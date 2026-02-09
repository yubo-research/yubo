import json
from pathlib import Path

import torch

from rl.algos.checkpointing import (
    CheckpointManager,
    append_jsonl,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_manager_paths(tmp_path):
    manager = CheckpointManager(exp_dir=Path(tmp_path))
    ckpt_path, latest_path = manager.paths(12)

    assert ckpt_path == Path(tmp_path) / "checkpoints" / "iter_000012.pt"
    assert latest_path == Path(tmp_path) / "checkpoint_last.pt"


def test_checkpoint_manager_roundtrip(tmp_path):
    manager = CheckpointManager(exp_dir=Path(tmp_path))
    state = {"iteration": 5, "value": torch.tensor([2.0])}
    ckpt_path, latest_path = manager.save_both(state, iteration=5)

    assert ckpt_path.exists()
    assert latest_path.exists()
    loaded = load_checkpoint(ckpt_path, torch.device("cpu"))
    assert loaded["iteration"] == 5
    assert float(loaded["value"][0]) == 2.0


def test_append_jsonl_writes_one_sorted_record(tmp_path):
    out = Path(tmp_path) / "metrics.jsonl"
    append_jsonl(out, {"z": 3, "a": 1})

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"a": 1, "z": 3}
    assert lines[0].startswith('{"a": 1')


def test_save_checkpoint_roundtrip(tmp_path):
    out = Path(tmp_path) / "manual.pt"
    state = {"step": 7, "weights": torch.tensor([1.0, 2.0])}
    save_checkpoint(out, state)

    loaded = load_checkpoint(out, torch.device("cpu"))
    assert loaded["step"] == 7
    assert torch.equal(loaded["weights"], torch.tensor([1.0, 2.0]))
