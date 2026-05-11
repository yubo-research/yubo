from typing import Any, Callable


_ATARI_DM_BINDINGS = None
_ATARI_DM_BINDINGS_LOADER: Callable[[], Any] | None = None


def register_atari_dm_bindings_loader(loader: Callable[[], Any]) -> None:
    global _ATARI_DM_BINDINGS_LOADER, _ATARI_DM_BINDINGS
    _ATARI_DM_BINDINGS_LOADER = loader
    _ATARI_DM_BINDINGS = None


def get_atari_dm_bindings():
    global _ATARI_DM_BINDINGS
    if _ATARI_DM_BINDINGS is None:
        if _ATARI_DM_BINDINGS_LOADER is None:
            raise RuntimeError("Atari/DM bindings are not registered. Call problems.env_conf_backends.register_with_env_conf() before using Atari/DM env tags.")
        _ATARI_DM_BINDINGS = _ATARI_DM_BINDINGS_LOADER()
    return _ATARI_DM_BINDINGS
