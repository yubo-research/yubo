from __future__ import annotations

from typing import Any, Callable

_GET_ENV_CONF: Callable[..., Any] | None = None


def register_get_env_conf(fn: Callable[..., Any]) -> None:
    global _GET_ENV_CONF
    _GET_ENV_CONF = fn


def get_env_conf_fn() -> Callable[..., Any]:
    return _GET_ENV_CONF
