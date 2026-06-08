"""SAC (TorchRL) public API."""

__all__ = [
    "SACCheckpointConfig",
    "SACCollectorConfig",
    "SACConfig",
    "SACEvalConfig",
    "SACLossConfig",
    "SACOptimConfig",
    "SACReplayBufferConfig",
    "SACTargetNetUpdaterConfig",
    "TrainResult",
    "register",
    "train_sac",
]


def __getattr__(name: str):
    import importlib

    if name in (
        "SACCheckpointConfig",
        "SACCollectorConfig",
        "SACConfig",
        "SACEvalConfig",
        "SACLossConfig",
        "SACOptimConfig",
        "SACReplayBufferConfig",
        "SACTargetNetUpdaterConfig",
        "TrainResult",
    ):
        m = importlib.import_module("rl.torchrl.sac.config")
        return getattr(m, name)
    if name in ("register", "train_sac"):
        m = importlib.import_module("rl.torchrl.sac.sac_train_loop")
        return getattr(m, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted(__all__)
