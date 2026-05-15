from rl.core import replay

ReplayBuffer = replay.NumpyReplayBuffer
TorchRLReplayBuffer = replay.TorchRLReplayBuffer
make_replay_buffer = replay.make_replay_buffer
resolve_replay_backend = replay.resolve_replay_backend


__all__ = [
    "ReplayBuffer",
    "TorchRLReplayBuffer",
    "make_replay_buffer",
    "resolve_replay_backend",
]
