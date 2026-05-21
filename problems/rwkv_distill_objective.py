from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from problems.pre_obj_stack import _log, _require_stack
from problems.pre_obj_subspace import _SubspaceParamCodec
from problems.pre_obj_validate import _build_validate, _ValidateContext


@dataclass(frozen=True)
class RWKVDistillObjectiveSpec:
    env_tag: str
    teacher_model_choice: str
    student_model_choice: str
    rwkv_type: str = "BaseRWKV"
    dtype: str | None = None
    generation_length: int = 256


class RWKVDistillObjective:
    """Teacher-forced KL distillation objective for quantized RWKV models."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._eval_count = 0
        stack = _require_stack()
        self.jax = stack.jax
        self.jnp = stack.jnp

        def _get(key, default):
            val = getattr(cfg, key, None)
            return default if val is None else val

        self.spec = RWKVDistillObjectiveSpec(
            env_tag=str(cfg.env_tag),
            teacher_model_choice=str(_get("distill_teacher_model_choice", "7g7B")),
            student_model_choice=str(_get("distill_student_model_choice", "7g7B")),
            rwkv_type=str(_get("pretrain_rwkv_type", "BaseRWKV")),
            dtype=_get("distill_dtype", None),
            generation_length=int(_get("distill_generation_length", _get("pretrain_generation_length", 256))),
        )

        _log(
            f"distill init env_tag={self.spec.env_tag} teacher={self.spec.teacher_model_choice} "
            f"student={self.spec.student_model_choice} generation_length={self.spec.generation_length}"
        )
        t0 = time.perf_counter()
        teacher = _load_model(stack.get_model, self.spec.teacher_model_choice, rwkv_type=self.spec.rwkv_type, dtype=self.spec.dtype)
        student = _load_model(stack.get_model, self.spec.student_model_choice, rwkv_type=self.spec.rwkv_type, dtype=self.spec.dtype)
        _log(f"teacher/student models ready dt={time.perf_counter() - t0:.2f}s")
        teacher_model, teacher_full, teacher_tokenizer = teacher
        teacher_config, teacher_params, teacher_scan_map, teacher_es_map = teacher_full

        student_model, student_full, student_tokenizer = student
        student_config, student_params, student_scan_map, student_es_map = student_full

        seed = int(getattr(cfg, "problem_seed", 0) or 0) + int(getattr(cfg, "seed_offset", 0) or 0)
        key = stack.jax.random.key(seed & 0xFFFFFFFF)

        self._teacher_validate = _build_validate(
            _ValidateContext(
                stack=stack,
                model=teacher_model,
                config=teacher_config,
                params_example=teacher_params,
                base_evo_keys=stack.simple_es_tree_key(teacher_params, stack.jax.random.fold_in(key, 1), teacher_scan_map),
                master_gen_key=stack.jax.random.fold_in(key, 2),
                tokenizer=teacher_tokenizer,
                legacy_tokenizer=stack.legacy_tokenizer_cls(),
                args=SimpleNamespace(
                    task="gsm8k",
                    generation_length=self.spec.generation_length,
                    parallel_validations=max(1, int(getattr(cfg, "num_envs", 1))),
                    validation_iterations=1,
                ),
            ),
            temperature=0.0,
            use_validation_set=True,
            noiser_cls=stack.noiser_cls,
            sigma=0.0,
        )
        self._codec = _SubspaceParamCodec(
            stack.jax,
            stack.jnp,
            student_params,
            es_map=student_es_map,
            dim=int(_get("distill_search_dim", _get("pretrain_search_dim", 4096))),
            delta_scale=float(_get("distill_delta_scale", _get("pretrain_delta_scale", 1.0))),
            seed=seed,
            lora_only=bool(_get("distill_lora_only", _get("pretrain_lora_only", True))),
            basis_max_leaves=_get("distill_basis_max_leaves", _get("pretrain_basis_max_leaves", None)),
        )
        self._student_model = student_model
        self._student_params = student_params
        self._student_scan_map = student_scan_map
        self._student_base_keys = stack.simple_es_tree_key(student_params, stack.jax.random.fold_in(key, 3), student_scan_map)
        self._student_teacher = teacher_model
        self._num_validation_samples = max(1, int(getattr(cfg, "num_envs", 1)))

    @property
    def dim(self) -> int:
        return self._codec.dim

    @property
    def x0(self) -> np.ndarray:
        return self._codec.x0

    @property
    def steps_per_episode(self) -> int:
        return 1

    @property
    def num_envs(self) -> int:
        return self._num_validation_samples

    def make_policy(self, x: np.ndarray):
        return SimpleNamespace(_rwkv_distill_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        params = self._codec.decode(np.asarray(x, dtype=np.float64))
        score = float(self._teacher_validate(params, int(seed)))
        if 0.0 <= score <= 1.0:
            se = float((score * (1.0 - score) / self._num_validation_samples) ** 0.5)
        else:
            se = 0.0
        self._eval_count += 1
        return score, se

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        from problems.pre_obj_vector_helpers import evaluate_many_serial

        return evaluate_many_serial(self.evaluate, x_batch, seed=seed)

    def configure_embedding(self, num_probes: int) -> None:
        from problems.pre_obj_vector_helpers import configure_embedding_indices

        self._embed_indices = configure_embedding_indices(self.dim, num_probes)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        from problems.pre_obj_vector_helpers import embed_many_with_indices

        if getattr(self, "_embed_indices", None) is None:
            self.configure_embedding(64)
        return embed_many_with_indices(x_batch, self._embed_indices)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        from problems.pre_obj_vector_helpers import sample_vector_noise

        return sample_vector_noise(dim=self.dim, seed=int(seed), num_dim_target=num_dim_target, num_module_target=num_module_target)


def _load_model(get_model, model_choice: str, *, rwkv_type: str, dtype: str | None):
    return get_model(model_choice, rwkv_type=rwkv_type, verbose=True, dtype=dtype)
