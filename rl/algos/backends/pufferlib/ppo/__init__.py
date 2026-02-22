"""PufferLib PPO backend package."""

from rl.algos.backends.pufferlib.ppo.api import (
    PufferPPOConfig,
    TrainResult,
    register,
    train_ppo_puffer,
)

__all__ = [
    "PufferPPOConfig",
    "TrainResult",
    "register",
    "train_ppo_puffer",
]
