from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from llm.registry import LLMEnvSpec, LLMPolicySpec, resolve_llm_env, resolve_llm_policy
from ops.uhd_config import UHDConfig


_TEXT_TAG_PREFIX = "llm:"
_RUNTIME_ERROR = (
    "Real UHD text runs require the CUDA text runtime: ray, vllm, transformers, "
    "torch, peft, accelerate, and safetensors. Run admin/setup-hyperscalees.sh on "
    "the CUDA machine, then launch with ./ops/exp_uhd.py from that environment."
)


@dataclass(frozen=True)
class TextSpec:
    env: LLMEnvSpec
    policy: LLMPolicySpec


def is_text_env(env_tag: str) -> bool:
    return str(env_tag).startswith(_TEXT_TAG_PREFIX)


def resolve_text_spec(env_tag: str, policy_tag: str | None) -> TextSpec:
    if not is_text_env(env_tag):
        raise ValueError(f"Unsupported text env_tag: {env_tag!r}. Expected prefix {_TEXT_TAG_PREFIX!r}.")
    if policy_tag is None:
        raise ValueError("UHD text objectives require policy_tag.")
    return TextSpec(env=resolve_llm_env(str(env_tag)), policy=resolve_llm_policy(str(policy_tag)))


class _PromptBatchCache:
    def __init__(self, max_size: int = 64) -> None:
        self._max_size = max(1, int(max_size))
        self._items: OrderedDict[int, tuple[list[str], list[Any]]] = OrderedDict()

    def get_or_create(self, key: int, create) -> tuple[list[str], list[Any]]:
        key = int(key)
        if key in self._items:
            self._items.move_to_end(key)
            prompts, answers = self._items[key]
            return list(prompts), list(answers)
        prompts, answers = create()
        self._items[key] = (list(prompts), list(answers))
        if len(self._items) > self._max_size:
            self._items.popitem(last=False)
        return list(prompts), list(answers)


class _LoraSubspaceCodec:
    """Sparse low-dimensional coordinates over a PEFT LoRA adapter state."""

    def __init__(
        self,
        template,
        *,
        dim: int,
        delta_scale: float,
        seed: int,
        basis_max_tensors: int | None,
    ) -> None:
        self.template = template
        self.dim = int(dim)
        self.delta_scale = float(delta_scale)

        leaves = [(name, value) for name, value in template.state_dict.items() if _is_search_tensor(name)]
        if not leaves:
            leaves = list(template.state_dict.items())
        if not leaves:
            raise ValueError("LoRA template has no trainable tensors for UHD text search.")

        sizes = np.asarray([int(tensor.numel()) for _, tensor in leaves], dtype=np.int64)
        valid = np.flatnonzero(sizes > 0)
        if valid.size == 0:
            raise ValueError("LoRA template tensors are all empty.")
        if basis_max_tensors is not None and int(basis_max_tensors) < valid.size:
            rng_for_tensors = np.random.default_rng(int(seed) ^ 0x9E3779B9)
            tensor_probs = sizes[valid].astype(np.float64)
            tensor_probs = tensor_probs / tensor_probs.sum()
            valid = np.sort(rng_for_tensors.choice(valid, size=int(basis_max_tensors), replace=False, p=tensor_probs).astype(np.int64))

        probs = sizes[valid].astype(np.float64)
        probs = probs / probs.sum()
        rng = np.random.default_rng(int(seed))
        self._names = tuple(name for name, _ in leaves)
        self._basis_tensor = rng.choice(valid, size=self.dim, replace=True, p=probs).astype(np.int64)
        self._basis_index = np.asarray([rng.integers(sizes[tensor_idx]) for tensor_idx in self._basis_tensor], dtype=np.int64)
        self._basis_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.dim).astype(np.float32)
        self.num_total_tensors = int(len(leaves))
        self.num_candidate_tensors = int(valid.size)
        self.num_candidate_params = int(sizes[valid].sum())
        self.num_basis_tensors = int(np.unique(self._basis_tensor).size)
        self.x0 = np.zeros((self.dim,), dtype=np.float64)

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        coeffs = np.asarray(x, dtype=np.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")
        state = {name: tensor.clone() for name, tensor in self.template.state_dict.items()}
        active = np.flatnonzero(coeffs != 0.0)
        if active.size == 0:
            return state

        import torch

        active_basis_tensor = self._basis_tensor[active]
        for tensor_idx in np.unique(active_basis_tensor):
            positions = active[np.flatnonzero(active_basis_tensor == tensor_idx)]
            name = self._names[int(tensor_idx)]
            flat = state[name].reshape(-1)
            idx = torch.as_tensor(self._basis_index[positions], dtype=torch.long, device=flat.device)
            values = torch.as_tensor(
                coeffs[positions] * self._basis_sign[positions] * self.delta_scale,
                dtype=flat.dtype,
                device=flat.device,
            )
            flat.index_add_(0, idx, values)
        return state


class TextObjective:
    """UHD vector objective for local text generation/scoring tasks.

    The objective is text; vLLM is only the inference backend. UHD optimizes a
    low-dimensional vector that is decoded into a sparse PEFT LoRA adapter.
    """

    def __init__(self, cfg: UHDConfig) -> None:
        self.cfg = cfg
        self.spec = resolve_text_spec(cfg.env_tag, cfg.policy_tag)
        self._dim = int(cfg.text_search_dim)
        self._x0 = np.zeros((self._dim,), dtype=np.float64)
        self._codec: _LoraSubspaceCodec | None = None
        self._pool = None
        self._tokenizer = None
        self._task = None
        self._adapter_root: Path | None = None
        self._sampling_kwargs = None
        self._batch_cache = _PromptBatchCache()
        self._embed_indices: np.ndarray | None = None
        self._eval_count = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def steps_per_episode(self) -> int:
        return int(self.cfg.prompt_batch_size)

    @property
    def eval_episodes(self) -> int:
        return 1

    def make_policy(self, x: np.ndarray):
        return SimpleNamespace(_text_uhd_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        runtime = self._ensure_runtime()
        prompts, answers = self._batch_for_seed(int(seed))
        adapter_path = self._materialize_adapter(np.asarray(x, dtype=np.float64), seed=int(seed))
        lora_name = f"text_{self._eval_count}"
        lora_id = 1 + int(hashlib.sha1(f"{seed}:{self._eval_count}".encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000
        lora_specs = [(lora_name, lora_id, str(adapter_path)) for _ in prompts]
        sampling = runtime.sampling_kwargs(
            tokenizer=self._tokenizer,
            temperature=float(self.cfg.temperature),
            seed=int(seed),
            max_tokens=int(self.cfg.max_tokens),
            n=int(self.cfg.samples_per_prompt),
        )
        args = SimpleNamespace(pass_at_k=bool(self.cfg.pass_at_k))

        try:
            fitnesses, _info, _logs = runtime.pool.generate_and_score(
                prompts=prompts,
                sampling_params_kwargs=sampling,
                lora_request_specs=lora_specs,
                task_obj=self._task,
                answers=answers,
                args=args,
            )
        finally:
            self._eval_count += 1
            shutil.rmtree(adapter_path, ignore_errors=True)

        values = np.asarray(fitnesses, dtype=np.float64)
        if values.size == 0:
            return 0.0, 0.0
        mu = float(np.mean(values))
        se = 0.0 if values.size < 2 else float(np.std(values, ddof=1) / np.sqrt(values.size))
        return mu, se

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        mus, ses = [], []
        for i, x in enumerate(np.asarray(x_batch, dtype=np.float64)):
            mu, se = self.evaluate(x, seed=int(seed) + i)
            mus.append(mu)
            ses.append(se)
        return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)

    def configure_embedding(self, num_probes: int) -> None:
        rng = np.random.default_rng(123)
        n = min(max(int(num_probes), 1), self.dim)
        self._embed_indices = rng.choice(self.dim, size=n, replace=False)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        if self._embed_indices is None:
            self.configure_embedding(64)
        assert self._embed_indices is not None
        return np.asarray(x_batch, dtype=np.float64)[:, self._embed_indices]

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        return _sample_vector_noise(
            dim=self.dim,
            seed=int(seed),
            num_dim_target=num_module_target if num_module_target is not None else num_dim_target,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._adapter_root is not None:
            shutil.rmtree(self._adapter_root, ignore_errors=True)
            self._adapter_root = None

    def _ensure_runtime(self):
        if self._pool is not None:
            assert self._sampling_kwargs is not None
            return SimpleNamespace(pool=self._pool, sampling_kwargs=self._sampling_kwargs)

        if self.cfg.hf_home:
            os.environ["HF_HOME"] = str(self.cfg.hf_home)
        _require_runtime()
        from transformers import AutoTokenizer

        from llm.engine_pool import EnginePoolConfig, VLLMEnginePool, sampling_kwargs
        from llm.lora import build_peft_lora_template
        from llm.tasks import build_task

        self._sampling_kwargs = sampling_kwargs
        base_seed = _base_seed(self.cfg)
        print(
            "UHD-Text: init start "
            f"env_tag={self.cfg.env_tag} model={self.spec.policy.model_name} dim={self.dim} "
            f"prompt_batch_size={self.cfg.prompt_batch_size}",
            flush=True,
        )
        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.policy.model_name, trust_remote_code=True)
        template = build_peft_lora_template(
            model_name=self.spec.policy.model_name,
            rank=int(self.spec.policy.lora_rank),
            alpha=int(self.spec.policy.lora_alpha),
        )
        self._codec = _LoraSubspaceCodec(
            template,
            dim=int(self.cfg.text_search_dim),
            delta_scale=float(self.cfg.text_delta_scale),
            seed=base_seed,
            basis_max_tensors=self.cfg.text_basis_max_tensors,
        )
        self._task = build_task(
            self.spec.env,
            batch_size=int(self.cfg.prompt_batch_size),
            seed=base_seed,
            max_tokens=int(self.cfg.max_tokens),
            dataset_size=self.cfg.sub_dataset_size,
            tokenizer=self._tokenizer,
            apply_chat_template=False,
        )
        tensor_parallel_size = int(self.cfg.tensor_parallel_size or self.spec.policy.tensor_parallel_size)
        num_engines = int(self.cfg.num_engines or max(1, int(self.cfg.num_gpus or tensor_parallel_size) // tensor_parallel_size))
        self._pool = VLLMEnginePool.launch(
            EnginePoolConfig(
                model_name=self.spec.policy.model_name,
                tensor_parallel_size=tensor_parallel_size,
                num_engines=num_engines,
                lora_rank=int(self.spec.policy.lora_rank),
                max_loras_per_engine=1,
                max_tokens=int(self.cfg.max_tokens),
                prompt_batch_size=int(self.cfg.prompt_batch_size),
            )
        )
        self._adapter_root = Path(_make_adapter_root())
        print(
            "UHD-Text: runtime ready "
            f"dt={time.perf_counter() - t0:.2f}s basis_tensors={self._codec.num_basis_tensors}/"
            f"{self._codec.num_candidate_tensors} candidate_params={self._codec.num_candidate_params}",
            flush=True,
        )
        assert self._sampling_kwargs is not None
        return SimpleNamespace(pool=self._pool, sampling_kwargs=self._sampling_kwargs)

    def _batch_for_seed(self, seed: int) -> tuple[list[str], list[Any]]:
        if self._task is None:
            raise RuntimeError("Text runtime has not been initialized.")
        return self._batch_cache.get_or_create(int(seed), self._task.get_batch)

    def _materialize_adapter(self, x: np.ndarray, *, seed: int) -> Path:
        if self._codec is None or self._adapter_root is None:
            raise RuntimeError("Text runtime has not been initialized.")
        adapter_path = self._adapter_root / f"eval_{self._eval_count}_{int(seed)}"
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        state = self._codec.decode(x)
        _write_lora_adapter(adapter_path, state, self._codec.template.config)
        return adapter_path


def _is_search_tensor(name: str) -> bool:
    # Keep the objective locally linear: fixed random A matrices, searchable B matrices.
    return ".lora_B." in str(name)


def _write_lora_adapter(adapter_path: Path, state: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Text UHD adapter materialization requires safetensors.") from exc

    with open(adapter_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)
    save_file({_adapter_tensor_name(name): tensor for name, tensor in state.items()}, str(adapter_path / "adapter_model.safetensors"))


def _adapter_tensor_name(name: str) -> str:
    return str(name).replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def _require_runtime() -> None:
    missing = [
        module
        for module in ("accelerate", "peft", "ray", "safetensors", "torch", "transformers", "vllm")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(f"{_RUNTIME_ERROR} Missing: {', '.join(sorted(missing))}.")


def _make_adapter_root() -> str:
    parent = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    return tempfile.mkdtemp(prefix="yubo_text_uhd_", dir=parent)


def _base_seed(cfg: UHDConfig) -> int:
    seed = cfg.noise_seed_0
    if seed is None:
        seed = cfg.problem_seed
    if seed is None:
        seed = 0
    return int(seed) + int(cfg.seed_offset)


def _sample_vector_noise(*, dim: int, seed: int, num_dim_target: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    dim = int(dim)
    if num_dim_target is None:
        return rng.standard_normal(dim).astype(np.float64)
    target = float(num_dim_target)
    if target <= 0:
        raise ValueError("perturb target must be > 0.")
    if 0 < target < 1:
        mask = rng.random(dim) < target
        if not np.any(mask):
            mask[int(rng.integers(dim))] = True
        noise = rng.standard_normal(dim).astype(np.float64)
        noise[~mask] = 0.0
        return noise
    k = min(max(int(target), 1), dim)
    idx = rng.choice(dim, size=k, replace=False)
    noise = np.zeros(dim, dtype=np.float64)
    noise[idx] = rng.standard_normal(k)
    return noise


__all__ = ["TextObjective", "TextSpec", "is_text_env", "resolve_text_spec"]
