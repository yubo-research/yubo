import json
from pathlib import Path

import torch

from rl.algos.checkpointing import (
    CheckpointManager,
    append_jsonl,
    load_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint,
)


def test_checkpoint_manager_paths(tmp_path):
    manager = CheckpointManager(exp_dir=Path(tmp_path))
    ckpt_path, latest_path = manager.paths(12)

    assert ckpt_path == Path(tmp_path) / "checkpoints" / "iter_000012.pt"
    assert latest_path == Path(tmp_path) / "checkpoints" / "checkpoint_last.pt"


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


def test_resolve_checkpoint_path_returns_existing(tmp_path):
    ckpt = Path(tmp_path) / "checkpoints" / "checkpoint_last.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("x", encoding="utf-8")
    assert resolve_checkpoint_path(ckpt) == ckpt


def test_resolve_checkpoint_path_finds_alternate_old_layout(tmp_path):
    exp_dir = Path(tmp_path) / "exp"
    old = exp_dir / "checkpoint_last.pt"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("x", encoding="utf-8")
    query = exp_dir / "checkpoints" / "checkpoint_last.pt"
    assert resolve_checkpoint_path(query) == old


def test_resolve_checkpoint_path_finds_alternate_new_layout(tmp_path):
    exp_dir = Path(tmp_path) / "exp"
    new = exp_dir / "checkpoints" / "checkpoint_last.pt"
    new.parent.mkdir(parents=True, exist_ok=True)
    new.write_text("x", encoding="utf-8")
    query = exp_dir / "checkpoint_last.pt"
    assert resolve_checkpoint_path(query) == new


def test_resolve_checkpoint_path_raises_if_missing(tmp_path):
    missing = Path(tmp_path) / "exp" / "checkpoints" / "checkpoint_last.pt"
    try:
        resolve_checkpoint_path(missing)
    except FileNotFoundError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing checkpoint.")
