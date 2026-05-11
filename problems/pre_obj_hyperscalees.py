from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from problems.pre_obj_specs import HyperscaleESPretrainSpec, resolve_hyperscalees_pretrain_spec
from problems.pre_obj_stack import _load_hyperscalees_model, _log, _require_stack
from problems.pre_obj_subspace import _SubspaceParamCodec
from problems.pre_obj_validate import _build_validate, _ValidateContext
from problems.pre_obj_vector_helpers import (
    configure_embedding_indices,
    embed_many_with_indices,
    evaluate_many_serial,
    sample_vector_noise,
)


class HyperscaleESLLMVectorObjective:
    """Real HyperscaleES LLM generation/scoring objective for UHD optimizers."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.spec = resolve_hyperscalees_pretrain_spec(cfg.env_tag)
        stack = _require_stack()
        self.jax = stack.jax
        self.jnp = stack.jnp
        self._eval_count = 0

        if cfg.pretrain_rwkv_type is not None:
            self.spec = HyperscaleESPretrainSpec(
                env_tag=self.spec.env_tag,
                task=self.spec.task,
                model_choice=self.spec.model_choice,
                thinking_length=self.spec.thinking_length,
                answer_length=self.spec.answer_length,
                rwkv_type=str(cfg.pretrain_rwkv_type),
                dtype=self.spec.dtype,
            )
        generation_length = self.spec.generation_length
        if cfg.pretrain_generation_length is not None:
            generation_length = int(cfg.pretrain_generation_length)

        _log(f"init start env_tag={cfg.env_tag} model={self.spec.model_choice} rwkv_type={self.spec.rwkv_type} generation_length={generation_length}")
        t0 = time.perf_counter()
        rwkv, full_params, tokenizer = _load_hyperscalees_model(stack.get_model, self.spec)
        _log(f"model loaded dt={time.perf_counter() - t0:.2f}s")
        config, params, scan_map, es_map = full_params
        seed = (0 if cfg.problem_seed is None else int(cfg.problem_seed)) + int(cfg.seed_offset)
        key = stack.jax.random.key(seed & 0xFFFFFFFF)
        _log("building ES key tree")
        t0 = time.perf_counter()
        base_evo_keys = stack.simple_es_tree_key(params, stack.jax.random.fold_in(key, 1), scan_map)
        _log(f"ES key tree ready dt={time.perf_counter() - t0:.2f}s")
        legacy_tokenizer = stack.legacy_tokenizer_cls() if self.spec.model_choice.startswith("7") else tokenizer

        t0 = time.perf_counter()
        self._codec = _SubspaceParamCodec(
            stack.jax,
            stack.jnp,
            params,
            es_map=es_map,
            dim=int(cfg.pretrain_search_dim),
            delta_scale=float(cfg.pretrain_delta_scale),
            seed=seed,
            lora_only=bool(cfg.pretrain_lora_only),
            basis_max_leaves=cfg.pretrain_basis_max_leaves,
        )
        _log(
            "subspace ready "
            f"dt={time.perf_counter() - t0:.2f}s dim={self._codec.dim} "
            f"candidate_leaves={self._codec.num_candidate_leaves}/{self._codec.num_total_leaves} "
            f"basis_leaves={self._codec.num_basis_leaves} "
            f"candidate_params={self._codec.num_candidate_params}"
        )
        args = SimpleNamespace(
            task=self.spec.task,
            generation_length=generation_length,
            parallel_validations=max(1, int(cfg.num_envs)),
            validation_iterations=1,
        )
        self._num_validation_samples = int(args.parallel_validations) * int(args.validation_iterations)
        _log(f"building validation task={args.task} parallel_validations={args.parallel_validations} validation_iterations={args.validation_iterations}")
        t0 = time.perf_counter()
        self._validate = _build_validate(
            _ValidateContext(
                stack=stack,
                model=rwkv,
                config=config,
                params_example=params,
                base_evo_keys=base_evo_keys,
                master_gen_key=stack.jax.random.fold_in(key, 2),
                tokenizer=tokenizer,
                legacy_tokenizer=legacy_tokenizer,
                args=args,
            ),
            temperature=0.0,
            use_validation_set=True,
            noiser_cls=stack.noiser_cls,
            sigma=0.0,
        )
        _log(f"validation ready dt={time.perf_counter() - t0:.2f}s")
        self._embed_indices: np.ndarray | None = None

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
        return SimpleNamespace(_hyperscalees_pretrain_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        trace_eval = self._eval_count < 5
        if trace_eval:
            _log(f"eval start eval={self._eval_count} seed={int(seed)} active_coeffs={int(np.count_nonzero(x_arr))}")
        t0 = time.perf_counter()
        params = self._codec.decode(x_arr)
        score = float(self._validate(params, int(seed)))
        if trace_eval:
            _log(f"eval done eval={self._eval_count} dt={time.perf_counter() - t0:.2f}s score={score:.6f}")
        self._eval_count += 1
        if 0.0 <= score <= 1.0 and self._num_validation_samples > 0:
            se = float((score * (1.0 - score) / self._num_validation_samples) ** 0.5)
        else:
            se = 0.0
        return score, se

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return evaluate_many_serial(self.evaluate, x_batch, seed=seed)

    def configure_embedding(self, num_probes: int) -> None:
        self._embed_indices = configure_embedding_indices(self.dim, num_probes)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        if self._embed_indices is None:
            self.configure_embedding(64)
        assert self._embed_indices is not None
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
        target = num_module_target if num_module_target is not None else num_dim_target
        return sample_vector_noise(dim=self.dim, seed=int(seed), num_dim_target=target)
