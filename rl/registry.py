import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    config_cls: type
    train_fn: Callable


_ALGOS: dict[str, AlgoSpec] = {}
_LAZY: dict[str, str] = {}


def _normalize_algo_name(name: str) -> str:
    key = str(name).strip()
    if not key:
        raise ValueError("Algorithm name cannot be empty.")
    return key


def register_algo(name: str, config_cls: type, train_fn: Callable) -> None:
    key = _normalize_algo_name(name)
    if key in _ALGOS:
        raise ValueError(f"Algorithm '{key}' already registered.")
    _ALGOS[key] = AlgoSpec(name=key, config_cls=config_cls, train_fn=train_fn)


def register_algo_lazy(name: str, module_path: str) -> None:
    key = _normalize_algo_name(name)
    if key in _LAZY:
        raise ValueError(f"Lazy loader for '{key}' already registered.")
    _LAZY[key] = module_path


def _maybe_load_lazy(key: str) -> None:
    if key not in _ALGOS and key in _LAZY:
        importlib.import_module(_LAZY[key]).register()


def get_algo(name: str) -> AlgoSpec:
    key = _normalize_algo_name(name)
    _maybe_load_lazy(key)
    spec = _ALGOS.get(key)
    if spec is not None:
        return spec
    raise ValueError(f"Unknown algorithm '{key}'. Available: {available_algos()}")


def available_algos() -> list[str]:
    names = set(_ALGOS)
    names.update(_LAZY)
    return sorted(names)
