import torch

from rl.algos.checkpoint_helpers import (
    _checkpoint_paths,
    _load_checkpoint,
    _save_checkpoint,
)


def test_checkpoint_helpers_roundtrip(tmp_path):
    ckpt_path, latest_path = _checkpoint_paths(tmp_path, 3)
    state = {"iteration": 3, "log_std": torch.tensor([1.0])}
    _save_checkpoint(ckpt_path, state)
    _save_checkpoint(latest_path, state)
    loaded = _load_checkpoint(ckpt_path, torch.device("cpu"))
    assert loaded["iteration"] == 3
    assert float(loaded["log_std"][0]) == 1.0
