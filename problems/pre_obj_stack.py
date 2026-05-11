from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from typing import Any

from problems.pre_obj_specs import HyperscaleESPretrainSpec


_STACK_ERROR = (
    "Real HyperscaleES pretraining UHD requires the separate HyperscaleES environment. "
    "Run admin/setup-hyperscalees.sh first, then use the plain python CLI from that environment."
)


@dataclass(frozen=True)
class _HyperscaleESStack:
    jax: Any
    jnp: Any
    simple_es_tree_key: Any
    get_model: Any
    legacy_tokenizer_cls: Any
    noiser_cls: Any
    all_tasks: Any
    validation_tasks: Any


def _require_stack() -> _HyperscaleESStack:
    try:
        import jax
        import jax.numpy as jnp
        from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks
        from hyperscalees.models.common import simple_es_tree_key
        from hyperscalees.models.llm.auto import get_model
        from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
        from hyperscalees.noiser.base_noiser import Noiser
    except ImportError as exc:
        raise ImportError(_STACK_ERROR) from exc
    return _HyperscaleESStack(
        jax=jax,
        jnp=jnp,
        simple_es_tree_key=simple_es_tree_key,
        get_model=get_model,
        legacy_tokenizer_cls=LegacyWorldTokenizer,
        noiser_cls=Noiser,
        all_tasks=all_tasks,
        validation_tasks=validation_tasks,
    )


def _log(message: str) -> None:
    print(f"UHD-HyperscaleES: {message}", flush=True)


def _load_hyperscalees_model(get_model, spec: HyperscaleESPretrainSpec):
    try:
        return get_model(
            spec.model_choice,
            rwkv_type=spec.rwkv_type,
            verbose=True,
            dtype=spec.dtype,
        )
    except (pickle.UnpicklingError, EOFError):
        warnings.warn(
            "HyperscaleES converted model cache appears corrupt; retrying with reload_cache=True. "
            "If this repeats, remove ~/.cache/yubo/hyperscalees/hyperscalees_cache for this model and check disk space.",
            RuntimeWarning,
            stacklevel=2,
        )
        return get_model(
            spec.model_choice,
            rwkv_type=spec.rwkv_type,
            verbose=True,
            dtype=spec.dtype,
            reload_cache=True,
        )
