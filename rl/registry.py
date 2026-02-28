import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    config_cls: type
    train_fn: Callable


_ALGOS: dict[str, AlgoSpec] = {}
_LAZY: dict[str, str] = {}  # name -> module path
_BACKEND_BINDINGS: dict[tuple[str, str], str] = {}  # (algo_name, backend_name) -> implementation algo name


def register_algo(name: str, config_cls: type, train_fn: Callable) -> None:
    if name in _ALGOS:
        raise ValueError(f"Algorithm '{name}' already registered.")
    _ALGOS[name] = AlgoSpec(name=name, config_cls=config_cls, train_fn=train_fn)


def register_algo_lazy(name: str, module_path: str) -> None:
    if name in _LAZY:
        raise ValueError(f"Lazy loader for '{name}' already registered.")
    _LAZY[name] = module_path


def _normalize_backend_name(backend: str) -> str:
    key = str(backend).strip().lower()
    if not key:
        raise ValueError("Backend name cannot be empty.")
    return key


def register_algo_backend(algo: str, backend: str, implementation: str) -> None:
    algo_key = str(algo).strip()
    impl_key = str(implementation).strip()
    backend_key = _normalize_backend_name(backend)
    if not algo_key:
        raise ValueError("Algorithm name cannot be empty.")
    if not impl_key:
        raise ValueError("Implementation algorithm name cannot be empty.")
    key = (algo_key, backend_key)
    existing = _BACKEND_BINDINGS.get(key)
    if existing is not None and existing != impl_key:
        raise ValueError(f"Backend binding for algorithm '{algo_key}' and backend '{backend_key}' already points to '{existing}'.")
    _BACKEND_BINDINGS[key] = impl_key


def resolve_algo_name(name: str, backend: str | None = None) -> str:
    algo_key = str(name).strip()
    if not algo_key:
        raise ValueError("Algorithm name cannot be empty.")
    if backend is None:
        return algo_key
    backend_key = _normalize_backend_name(backend)
    key = (algo_key, backend_key)
    if key in _BACKEND_BINDINGS:
        return _BACKEND_BINDINGS[key]
    known_backends = sorted(b for a, b in _BACKEND_BINDINGS if a == algo_key)
    if known_backends:
        raise ValueError(f"Unknown backend '{backend_key}' for algorithm '{algo_key}'. Available backends: {known_backends}")
    # Allow pre-registration parse paths (for config catalog checks) to proceed.
    # Runtime calls that need a real implementation still go through get_algo().
    return algo_key


def get_algo(name: str, *, backend: str | None = None) -> AlgoSpec:
    resolved_name = resolve_algo_name(name, backend=backend)
    if resolved_name not in _ALGOS and resolved_name in _LAZY:
        importlib.import_module(_LAZY[resolved_name]).register()
    if resolved_name not in _ALGOS:
        raise ValueError(f"Unknown algorithm '{resolved_name}'. Available: {sorted(set(_ALGOS) | set(_LAZY))}")
    return _ALGOS[resolved_name]


def available_algos() -> list[str]:
    return sorted(set(_ALGOS) | set(_LAZY))


def available_backend_bindings() -> dict[tuple[str, str], str]:
    return dict(_BACKEND_BINDINGS)
