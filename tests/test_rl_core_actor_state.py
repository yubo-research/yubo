import numpy as np
import pytest
import torch
import torch.nn as nn

from rl.core.actor_state import (
    load,
    snap,
    using,
)


def _make_modules():
    backbone = nn.Linear(3, 4)
    head = nn.Linear(4, 2)
    log_std = nn.Parameter(torch.zeros(2))
    return backbone, head, log_std


def test_capture_snapshot_tensor_and_numpy_formats():
    backbone, head, log_std = _make_modules()
    snap_tensor = snap(
        backbone,
        head,
        log_std=log_std,
        state_to_cpu=True,
        log_std_to_cpu=True,
        log_std_format="tensor",
    )
    assert snap_tensor["backbone"]["weight"].device.type == "cpu"
    assert snap_tensor["head"]["bias"].device.type == "cpu"
    assert isinstance(snap_tensor["log_std"], torch.Tensor)

    snap_numpy = snap(
        backbone,
        head,
        log_std=log_std,
        state_to_cpu=False,
        log_std_to_cpu=True,
        log_std_format="numpy",
    )
    assert isinstance(snap_numpy["log_std"], np.ndarray)


def test_restore_snapshot_roundtrip():
    backbone, head, log_std = _make_modules()
    snapshot = snap(
        backbone,
        head,
        log_std=log_std,
        log_std_format="numpy",
    )
    with torch.no_grad():
        backbone.weight.add_(1.0)
        head.bias.sub_(2.0)
        log_std.add_(0.5)

    load(
        backbone,
        head,
        snapshot,
        log_std=log_std,
        device=torch.device("cpu"),
    )
    assert torch.equal(backbone.state_dict()["weight"], snapshot["backbone"]["weight"])
    assert torch.equal(head.state_dict()["bias"], snapshot["head"]["bias"])
    assert torch.allclose(log_std, torch.as_tensor(snapshot["log_std"]))


def test_use_snapshot_restores_previous_state():
    backbone, head, log_std = _make_modules()
    original = snap(
        backbone,
        head,
        log_std=log_std,
        log_std_format="numpy",
    )
    replacement = snap(
        backbone,
        head,
        log_std=log_std,
        log_std_format="numpy",
    )
    replacement["backbone"]["weight"] = replacement["backbone"]["weight"] + 3.0
    replacement["head"]["bias"] = replacement["head"]["bias"] - 1.0
    replacement["log_std"] = replacement["log_std"] + 0.75

    with using(
        backbone,
        head,
        replacement,
        log_std=log_std,
        device=torch.device("cpu"),
        log_std_format="numpy",
    ):
        inside = snap(
            backbone,
            head,
            log_std=log_std,
            log_std_format="numpy",
        )
        assert torch.equal(inside["backbone"]["weight"], replacement["backbone"]["weight"])
        assert torch.equal(inside["head"]["bias"], replacement["head"]["bias"])
        assert np.allclose(inside["log_std"], replacement["log_std"])

    after = snap(
        backbone,
        head,
        log_std=log_std,
        log_std_format="numpy",
    )
    assert torch.equal(after["backbone"]["weight"], original["backbone"]["weight"])
    assert torch.equal(after["head"]["bias"], original["head"]["bias"])
    assert np.allclose(after["log_std"], original["log_std"])


def test_capture_snapshot_invalid_log_std_format():
    backbone, head, log_std = _make_modules()
    with pytest.raises(ValueError, match="log_std_format"):
        snap(
            backbone,
            head,
            log_std=log_std,
            log_std_format="bad",
        )
