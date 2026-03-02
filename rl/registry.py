import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    config_cls: type
    train_fn: Callable


# (algo_name, backend_name|None) -> spec/module
_ALGOS: dict[tuple[str, str | None], AlgoSpec] = {}
_LAZY: dict[tuple[str, str | None], str] = {}
_BACKEND_BINDINGS: dict[tuple[str, str], str] = {}  # (algo_name, backend_name) -> implementation algo name


def _normalize_algo_name(name: str) -> str:
    key = str(name).strip()
    if not key:
        raise ValueError("Algorithm name cannot be empty.")
    return key


def _algo_key(name: str, backend: str | None = None) -> tuple[str, str | None]:
    algo_key = _normalize_algo_name(name)
    backend_key = None if backend is None else _normalize_backend_name(backend)
    return algo_key, backend_key


def _normalize_backend_name(backend: str) -> str:
    key = str(backend).strip().lower()
    if not key:
        raise ValueError("Backend name cannot be empty.")
    return key


def register_algo(name: str, config_cls: type, train_fn: Callable, *, backend: str | None = None) -> None:
    algo_key, backend_key = _algo_key(name, backend=backend)
    key = (algo_key, backend_key)
    if key in _ALGOS:
        if backend_key is None:
            raise ValueError(f"Algorithm '{algo_key}' already registered.")
        raise ValueError(f"Algorithm '{algo_key}' already registered for backend '{backend_key}'.")
    _ALGOS[key] = AlgoSpec(name=algo_key, config_cls=config_cls, train_fn=train_fn)


def register_algo_lazy(name: str, module_path: str, *, backend: str | None = None) -> None:
    algo_key, backend_key = _algo_key(name, backend=backend)
    key = (algo_key, backend_key)
    if key in _LAZY:
        if backend_key is None:
            raise ValueError(f"Lazy loader for '{algo_key}' already registered.")
        raise ValueError(f"Lazy loader for '{algo_key}' already registered for backend '{backend_key}'.")
    _LAZY[key] = module_path


def register_algo_backend(algo: str, backend: str, implementation: str) -> None:
    algo_key = _normalize_algo_name(algo)
    impl_key = _normalize_algo_name(implementation)
    backend_key = _normalize_backend_name(backend)
    key = (algo_key, backend_key)
    existing = _BACKEND_BINDINGS.get(key)
    if existing is not None and existing != impl_key:
        raise ValueError(f"Backend binding for algorithm '{algo_key}' and backend '{backend_key}' already points to '{existing}'.")
    _BACKEND_BINDINGS[key] = impl_key


def _known_backends(algo_key: str) -> list[str]:
    found = {backend for name, backend in _BACKEND_BINDINGS if name == algo_key}
    found.update(backend for name, backend in _ALGOS if name == algo_key and backend is not None)
    found.update(backend for name, backend in _LAZY if name == algo_key and backend is not None)
    return sorted(found)


def resolve_algo_name(name: str, backend: str | None = None) -> str:
    algo_key = _normalize_algo_name(name)
    if backend is None:
        return algo_key
    backend_key = _normalize_backend_name(backend)
    binding_key = (algo_key, backend_key)
    if binding_key in _BACKEND_BINDINGS:
        return _BACKEND_BINDINGS[binding_key]
    if (algo_key, backend_key) in _ALGOS or (algo_key, backend_key) in _LAZY:
        return algo_key
    known_backends = _known_backends(algo_key)
    if known_backends:
        raise ValueError(f"Unknown backend '{backend_key}' for algorithm '{algo_key}'. Available backends: {known_backends}")
    # Allow pre-registration parse paths (for config catalog checks) to proceed.
    # Runtime calls that need a real implementation still go through get_algo().
    return algo_key


def _maybe_load_lazy(key: tuple[str, str | None]) -> None:
    if key not in _ALGOS and key in _LAZY:
        importlib.import_module(_LAZY[key]).register()


def get_algo(name: str, *, backend: str | None = None) -> AlgoSpec:
    resolved_name = resolve_algo_name(name, backend=backend)
    if backend is None:
        candidate_keys = [(resolved_name, None)]
    else:
        backend_key = _normalize_backend_name(backend)
        candidate_keys = [
            (resolved_name, backend_key),
            (resolved_name, None),
        ]

    for key in candidate_keys:
        _maybe_load_lazy(key)
    for key in candidate_keys:
        spec = _ALGOS.get(key)
        if spec is not None:
            return spec

    raise ValueError(f"Unknown algorithm '{resolved_name}'. Available: {available_algos()}")


def available_algos() -> list[str]:
    names = {name for name, _backend in _ALGOS}
    names.update(name for name, _backend in _LAZY)
    names.update(name for name, _backend in _BACKEND_BINDINGS)
    names.update(implementation for implementation in _BACKEND_BINDINGS.values())
    return sorted(names)


def available_backend_bindings() -> dict[tuple[str, str], str]:
    return dict(_BACKEND_BINDINGS)
