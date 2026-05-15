from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "_build_training_components": "_train_run_build_components",
    "_init_run_artifacts": "_train_run_init_artifacts",
    "_init_runtime": "_train_run_init_runtime",
    "_log_header": "_train_run_log_header",
    "train_sac_puffer_impl": "train_sac_puffer_impl",
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    mod = importlib.import_module("rl.pufferlib.sac.sac_puffer_train_run_orchestrate")
    return getattr(mod, target)


__all__ = list(_EXPORTS)
