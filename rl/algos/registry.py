from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    config_cls: type
    train_fn: Callable


_ALGOS: dict[str, AlgoSpec] = {}


def register_algo(name: str, config_cls: type, train_fn: Callable) -> None:
    if name in _ALGOS:
        raise ValueError(f"Algorithm '{name}' already registered.")
    _ALGOS[name] = AlgoSpec(name=name, config_cls=config_cls, train_fn=train_fn)


def get_algo(name: str) -> AlgoSpec:
    if name not in _ALGOS:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {sorted(_ALGOS)}")
    return _ALGOS[name]


def available_algos() -> list[str]:
    return sorted(_ALGOS)
