import pytest
import torch

from rl.precision import PrecisionController, normalize_precision_mode, resolve_amp_dtype


def test_normalize_precision_mode_valid_values():
    assert normalize_precision_mode(None) == "auto"
    assert normalize_precision_mode("fp32") == "fp32"
    assert normalize_precision_mode("BF16") == "bf16"


def test_normalize_precision_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="precision must be one of"):
        normalize_precision_mode("fp16")


def test_resolve_amp_dtype_auto_cpu_is_fp32():
    dtype = resolve_amp_dtype("auto", device=torch.device("cpu"))
    assert dtype is None


def test_resolve_amp_dtype_bf16_rejects_cpu():
    with pytest.raises(ValueError, match="precision='bf16' requested"):
        resolve_amp_dtype("bf16", device=torch.device("cpu"))


def test_precision_controller_demotes_auto_on_bf16_runtime_error():
    ctrl = PrecisionController(
        mode="auto",
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )
    changed = ctrl.maybe_demote_on_runtime_error(RuntimeError("BFloat16 is not supported"), component="test")
    assert changed is True
    assert ctrl.amp_dtype is None
    assert ctrl.fallback_used is True


def test_precision_controller_factory_and_label_fp32():
    ctrl = PrecisionController.from_config("fp32", device=torch.device("cpu"))
    assert ctrl.amp_dtype is None
    assert ctrl.resolved_label() == "fp32"


def test_precision_controller_autocast_nullcontext():
    ctrl = PrecisionController.from_config("fp32", device=torch.device("cpu"))
    with ctrl.autocast():
        x = 1 + 1
    assert x == 2


def test_precision_controller_label_bf16():
    ctrl = PrecisionController(
        mode="auto",
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )
    assert ctrl.resolved_label() == "bf16"
