from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

import torch

_VALID_PRECISION_MODES = {"auto", "fp32", "bf16"}


def normalize_precision_mode(mode: str | None) -> str:
    key = str(mode or "auto").strip().lower()
    if key not in _VALID_PRECISION_MODES:
        raise ValueError("precision must be one of: auto, fp32, bf16")
    return key


def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if fn is None:
        return False
    return bool(fn())


def _mps_supports_bf16() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        return False
    x = torch.randn((8, 8), device="mps", dtype=torch.float32)
    return _try_mps_bf16_probe(x)


def _try_mps_bf16_probe(x: torch.Tensor) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                z = (x @ x.t()).sum()
            _ = float(z.float().item())
        return True
    except Exception:
        return False


@lru_cache(maxsize=8)
def _supports_bf16(device_type: str) -> bool:
    key = str(device_type).strip().lower()
    if key == "cuda":
        return _cuda_supports_bf16()
    if key == "mps":
        return _mps_supports_bf16()
    if key == "cpu":
        return False
    return False


def resolve_amp_dtype(mode: str | None, *, device: torch.device) -> torch.dtype | None:
    normalized = normalize_precision_mode(mode)
    if normalized == "fp32":
        return None
    if normalized == "auto":
        if device.type in {"cuda", "mps"} and _supports_bf16(device.type):
            return torch.bfloat16
        return None
    if not _supports_bf16(device.type):
        raise ValueError(f"precision='bf16' requested but bf16 is unsupported on device '{device.type}'. Use precision='fp32' or precision='auto'.")
    return torch.bfloat16


def _is_bf16_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    patterns = ("bfloat16", "bf16", "not implemented for", "unsupported dtype", "doesn't support float64", "not supported on mps")
    return any((p in msg for p in patterns))


@dataclass
class PrecisionController:
    mode: str
    device: torch.device
    amp_dtype: torch.dtype | None
    fallback_used: bool = False

    @classmethod
    def from_config(cls, mode: str | None, *, device: torch.device) -> "PrecisionController":
        normalized = normalize_precision_mode(mode)
        amp_dtype = resolve_amp_dtype(normalized, device=device)
        return cls(mode=normalized, device=device, amp_dtype=amp_dtype)

    def resolved_label(self) -> str:
        if self.amp_dtype == torch.bfloat16:
            return "bf16"
        return "fp32"

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def maybe_demote_on_runtime_error(self, exc: BaseException, *, component: str) -> bool:
        if self.mode != "auto" or self.amp_dtype != torch.bfloat16:
            return False
        if not _is_bf16_runtime_error(exc):
            return False
        self.amp_dtype = None
        self.fallback_used = True
        print(f"[{component}] precision=auto bf16 path failed; falling back to fp32.", flush=True)
        return True


__all__ = ["PrecisionController", "normalize_precision_mode", "resolve_amp_dtype"]
