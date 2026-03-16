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
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Algorithm name cannot be empty.")
    return key


def register_algo(name: str, config_cls: type, train_fn: Callable) -> None:
    algo_key = _normalize_algo_name(name)
    if algo_key in _ALGOS:
        raise ValueError(f"Algorithm '{algo_key}' already registered.")
    _ALGOS[algo_key] = AlgoSpec(name=algo_key, config_cls=config_cls, train_fn=train_fn)


def register_algo_lazy(name: str, module_path: str) -> None:
    algo_key = _normalize_algo_name(name)
    if algo_key in _LAZY:
        raise ValueError(f"Lazy loader for '{algo_key}' already registered.")
    _LAZY[algo_key] = module_path


def _maybe_load_lazy(algo_key: str) -> None:
    if algo_key not in _ALGOS and algo_key in _LAZY:
        importlib.import_module(_LAZY[algo_key]).register()


def get_algo(name: str) -> AlgoSpec:
    algo_key = _normalize_algo_name(name)
    _maybe_load_lazy(algo_key)
    spec = _ALGOS.get(algo_key)
    if spec is not None:
        return spec
    raise ValueError(f"Unknown algorithm '{algo_key}'. Available: {available_algos()}")


def available_algos() -> list[str]:
    names = set(_ALGOS) | set(_LAZY)
    return sorted(names)
