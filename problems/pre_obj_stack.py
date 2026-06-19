from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from problems.pre_obj_specs import HyperscaleESPretrainSpec

_STACK_ERROR = (
    "Real HyperscaleES pretraining UHD requires the separate HyperscaleES environment. Run the Pixi setup task first, then use that Pixi environment."
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


def _new_legacy_tokenizer(stack: _HyperscaleESStack):
    try:
        return stack.legacy_tokenizer_cls()
    except FileNotFoundError as exc:
        from hyperscalees.models.llm.tokenizer import RWKV_TOKENIZER

        vocab = files("pyrwkv_tokenizer").joinpath("rwkv_vocab_v20230424.txt")
        if not vocab.is_file():
            raise FileNotFoundError(
                "HyperscaleES omitted rwkv_vocab_v20230424.txt from its installed package and the pyrwkv-tokenizer fallback is unavailable."
            ) from exc
        _log("using pyrwkv-tokenizer vocab fallback for upstream legacy tokenizer")
        tokenizer = stack.legacy_tokenizer_cls.__new__(stack.legacy_tokenizer_cls)
        tokenizer.tok = RWKV_TOKENIZER(str(vocab))
        return tokenizer


def _place_model_params(stack: _HyperscaleESStack, full_params):
    target = stack.jax.devices()[0]
    t0 = time.perf_counter()
    params = stack.jax.device_put(full_params.params, target)
    params = stack.jax.block_until_ready(params)
    _log(f"model params ready device={target} dt={time.perf_counter() - t0:.2f}s")
    return full_params._replace(params=params)


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
