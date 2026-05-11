from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from problems import pre_obj_vector_helpers as vector_helpers
from problems import text_obj_lora as lora
from problems import text_obj_runtime as runtime
from problems.text_obj_cache import _PromptBatchCache
from problems.text_obj_specs import resolve_text_spec


class TextObjective:
    """UHD vector objective for local text generation/scoring tasks."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.spec = resolve_text_spec(cfg.env_tag, cfg.policy_tag)
        self._dim = int(cfg.text_search_dim)
        self._x0 = np.zeros((self._dim,), dtype=np.float64)
        self._codec: lora._LoraSubspaceCodec | None = None
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
    def num_envs(self) -> int:
        return 1

    def make_policy(self, x: np.ndarray):
        return SimpleNamespace(_text_uhd_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        runtime = _ensure_runtime(self)
        prompts, answers = _batch_for_seed(self, int(seed))
        adapter_path = _materialize_adapter(self, np.asarray(x, dtype=np.float64), seed=int(seed))
        try:
            fitnesses = _generate_fitnesses(self, runtime, prompts, answers, adapter_path, int(seed))
        finally:
            self._eval_count += 1
            shutil.rmtree(adapter_path, ignore_errors=True)
        values = np.asarray(fitnesses, dtype=np.float64)
        if values.size == 0:
            return 0.0, 0.0
        return float(np.mean(values)), _standard_error(values)

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return vector_helpers.evaluate_many_serial(self.evaluate, x_batch, seed=seed)

    def configure_embedding(self, num_probes: int) -> None:
        self._embed_indices = vector_helpers.configure_embedding_indices(self.dim, num_probes)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        return vector_helpers.embed_many_with_indices(x_batch, _embedding_indices(self))

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        return vector_helpers.sample_vector_noise(
            dim=self.dim,
            seed=int(seed),
            num_dim_target=_noise_target(num_dim_target, num_module_target),
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._adapter_root is not None:
            shutil.rmtree(self._adapter_root, ignore_errors=True)
            self._adapter_root = None


def _ensure_runtime(obj: TextObjective):
    if obj._pool is not None:
        assert obj._sampling_kwargs is not None
        return SimpleNamespace(pool=obj._pool, sampling_kwargs=obj._sampling_kwargs)

    if obj.cfg.hf_home:
        os.environ["HF_HOME"] = str(obj.cfg.hf_home)
    runtime.require_runtime()
    from transformers import AutoTokenizer

    from llm.engine_pool import sampling_kwargs
    from llm.lora import build_peft_lora_template
    from llm.tasks import build_task

    obj._sampling_kwargs = sampling_kwargs
    seed = runtime.base_seed(obj.cfg)
    print(
        f"UHD-Text: init start env_tag={obj.cfg.env_tag} model={obj.spec.policy.model_name} dim={obj.dim} prompt_batch_size={obj.cfg.prompt_batch_size}",
        flush=True,
    )
    t0 = time.perf_counter()
    obj._tokenizer = AutoTokenizer.from_pretrained(obj.spec.policy.model_name, trust_remote_code=True)
    template = build_peft_lora_template(
        model_name=obj.spec.policy.model_name,
        rank=int(obj.spec.policy.lora_rank),
        alpha=int(obj.spec.policy.lora_alpha),
    )
    obj._codec = lora._LoraSubspaceCodec(
        template,
        dim=int(obj.cfg.text_search_dim),
        delta_scale=float(obj.cfg.text_delta_scale),
        seed=seed,
        basis_max_tensors=obj.cfg.text_basis_max_tensors,
    )
    obj._task = build_task(
        obj.spec.env,
        batch_size=int(obj.cfg.prompt_batch_size),
        seed=seed,
        max_tokens=int(obj.cfg.max_tokens),
        dataset_size=obj.cfg.sub_dataset_size,
        tokenizer=obj._tokenizer,
        apply_chat_template=False,
    )
    obj._pool = _launch_pool(obj.cfg, obj.spec.policy)
    obj._adapter_root = Path(runtime.make_adapter_root())
    print(
        "UHD-Text: runtime ready "
        f"dt={time.perf_counter() - t0:.2f}s basis_tensors={obj._codec.num_basis_tensors}/"
        f"{obj._codec.num_candidate_tensors} candidate_params={obj._codec.num_candidate_params}",
        flush=True,
    )
    assert obj._sampling_kwargs is not None
    return SimpleNamespace(pool=obj._pool, sampling_kwargs=obj._sampling_kwargs)


def _batch_for_seed(obj: TextObjective, seed: int) -> tuple[list[str], list[Any]]:
    if obj._task is None:
        raise RuntimeError("Text runtime has not been initialized.")
    return obj._batch_cache.get_or_create(int(seed), obj._task.get_batch)


def _materialize_adapter(obj: TextObjective, x: np.ndarray, *, seed: int) -> Path:
    if obj._codec is None or obj._adapter_root is None:
        raise RuntimeError("Text runtime has not been initialized.")
    adapter_path = obj._adapter_root / f"eval_{obj._eval_count}_{int(seed)}"
    if adapter_path.exists():
        shutil.rmtree(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    state = obj._codec.decode(x)
    lora._write_lora_adapter(adapter_path, state, obj._codec.template.config)
    return adapter_path


def _generate_fitnesses(obj: TextObjective, runtime, prompts: list[str], answers: list[Any], adapter_path: Path, seed: int) -> list[float]:
    sampling = runtime.sampling_kwargs(
        tokenizer=obj._tokenizer,
        temperature=float(obj.cfg.temperature),
        seed=int(seed),
        max_tokens=int(obj.cfg.max_tokens),
        n=int(obj.cfg.samples_per_prompt),
    )
    args = SimpleNamespace(pass_at_k=bool(obj.cfg.pass_at_k))
    lora_specs = _lora_specs(prompts, adapter_path, seed, obj._eval_count)
    fitnesses, _info, _logs = runtime.pool.generate_and_score(
        prompts=prompts,
        sampling_params_kwargs=sampling,
        lora_request_specs=lora_specs,
        task_obj=obj._task,
        answers=answers,
        args=args,
    )
    return fitnesses


def _embedding_indices(obj: TextObjective) -> np.ndarray:
    indices = obj._embed_indices
    if indices is not None:
        return indices
    obj.configure_embedding(64)
    if obj._embed_indices is None:
        raise RuntimeError("Text embedding indices were not configured.")
    return obj._embed_indices


def _noise_target(num_dim_target: float | None, num_module_target: float | None) -> float | None:
    return num_module_target if num_module_target is not None else num_dim_target


def _launch_pool(cfg: Any, policy: Any):
    from llm.engine_pool import EnginePoolConfig, VLLMEnginePool

    tensor_parallel_size = int(cfg.tensor_parallel_size or policy.tensor_parallel_size)
    num_gpus = int(cfg.num_gpus or tensor_parallel_size)
    num_engines = int(cfg.num_engines or max(1, num_gpus // tensor_parallel_size))
    return VLLMEnginePool.launch(
        EnginePoolConfig(
            model_name=policy.model_name,
            tensor_parallel_size=tensor_parallel_size,
            num_engines=num_engines,
            lora_rank=int(policy.lora_rank),
            max_loras_per_engine=1,
            max_tokens=int(cfg.max_tokens),
            prompt_batch_size=int(cfg.prompt_batch_size),
        )
    )


def _lora_specs(prompts: list[str], adapter_path: Path, seed: int, eval_count: int) -> list[tuple[str, int, str]]:
    lora_name = f"text_{eval_count}"
    lora_id = 1 + int(hashlib.sha1(f"{seed}:{eval_count}".encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000
    return [(lora_name, lora_id, str(adapter_path)) for _ in prompts]


def _standard_error(values: np.ndarray) -> float:
    return 0.0 if values.size < 2 else float(np.std(values, ddof=1) / np.sqrt(values.size))
