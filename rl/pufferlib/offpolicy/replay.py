from rl.core.replay import NumpyReplayBuffer as ReplayBuffer
from rl.core.replay import (
    TorchRLReplayBuffer,
    make_replay_buffer,
    resolve_replay_backend,
)

__all__ = [
    "ReplayBuffer",
    "TorchRLReplayBuffer",
    "make_replay_buffer",
    "resolve_replay_backend",
]
