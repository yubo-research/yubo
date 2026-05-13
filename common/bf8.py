from __future__ import annotations

import numpy as np


_BF8_MANT_BITS = 3
_BF8_EXP_BITS = 4
_BF8_EXP_BIAS = (1 << (_BF8_EXP_BITS - 1)) - 1
_BF8_EXP_MAX = (1 << _BF8_EXP_BITS) - 1
_BF8_FRAC_MAX = (1 << _BF8_MANT_BITS) - 1


def bf8_encode(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=np.uint8)
    flat_in = arr.reshape(-1)
    flat_out = out.reshape(-1)

    for i, x in enumerate(flat_in):
        flat_out[i] = _encode_scalar(float(x))
    return out


def bf8_decode(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    out = np.zeros(arr.shape, dtype=np.float32)
    flat_in = arr.reshape(-1)
    flat_out = out.reshape(-1)

    for i, x in enumerate(flat_in):
        flat_out[i] = _decode_scalar(int(x))
    return out


def bf8_encode_tree(tree):
    torch_tensor = _as_torch_tensor(tree)
    if torch_tensor is not None:
        encoded = bf8_encode(torch_tensor.detach().cpu().numpy())
        return _BF8TorchTensor(encoded=encoded, dtype=torch_tensor.dtype, device=torch_tensor.device)
    if isinstance(tree, np.ndarray):
        return bf8_encode(tree)
    if isinstance(tree, dict):
        return {k: bf8_encode_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [bf8_encode_tree(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(bf8_encode_tree(v) for v in tree)
    if tree is None:
        return None
    return bf8_encode(np.asarray(tree))


def bf8_decode_tree(tree):
    if isinstance(tree, _BF8TorchTensor):
        decoded = bf8_decode(tree.encoded)
        torch = _torch_or_none()
        assert torch is not None
        return torch.as_tensor(decoded, dtype=tree.dtype, device=tree.device)
    if isinstance(tree, np.ndarray):
        return bf8_decode(tree)
    if isinstance(tree, dict):
        return {k: bf8_decode_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [bf8_decode_tree(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(bf8_decode_tree(v) for v in tree)
    if tree is None:
        return None
    return bf8_decode(np.asarray(tree))


class _BF8TorchTensor:
    def __init__(self, *, encoded: np.ndarray, dtype, device) -> None:
        self.encoded = np.asarray(encoded, dtype=np.uint8)
        self.dtype = dtype
        self.device = device


def _torch_or_none():
    try:
        import torch
    except Exception:
        return None
    return torch


def _as_torch_tensor(value):
    torch = _torch_or_none()
    if torch is None:
        return None
    return value if isinstance(value, torch.Tensor) else None


def _encode_scalar(x: float) -> np.uint8:
    if np.isnan(x):
        return np.uint8(0x7F)
    sign = 1 if np.signbit(x) else 0
    ax = abs(x)
    if ax == 0.0:
        return np.uint8(sign << 7)
    if np.isinf(ax):
        return np.uint8((sign << 7) | (_BF8_EXP_MAX << _BF8_MANT_BITS))

    exp = int(np.floor(np.log2(ax)))
    mant = ax / (2.0**exp) - 1.0
    exp_biased = exp + _BF8_EXP_BIAS

    if exp_biased >= _BF8_EXP_MAX:
        return np.uint8((sign << 7) | (_BF8_EXP_MAX << _BF8_MANT_BITS))

    if exp_biased <= 0:
        sub = ax / (2.0 ** (1 - _BF8_EXP_BIAS))
        frac = int(np.rint(sub * (1 << _BF8_MANT_BITS)))
        frac = max(0, min(frac, _BF8_FRAC_MAX))
        return np.uint8((sign << 7) | frac)

    frac = int(np.rint(mant * (1 << _BF8_MANT_BITS)))
    if frac >= (1 << _BF8_MANT_BITS):
        frac = 0
        exp_biased += 1
        if exp_biased >= _BF8_EXP_MAX:
            return np.uint8((sign << 7) | (_BF8_EXP_MAX << _BF8_MANT_BITS))

    return np.uint8((sign << 7) | ((exp_biased & _BF8_EXP_MAX) << _BF8_MANT_BITS) | (frac & _BF8_FRAC_MAX))


def _decode_scalar(x: int) -> np.float32:
    sign = -1.0 if (x & 0x80) else 1.0
    exp = (x >> _BF8_MANT_BITS) & _BF8_EXP_MAX
    frac = x & _BF8_FRAC_MAX

    if exp == 0:
        if frac == 0:
            return np.float32(sign * 0.0)
        return np.float32(sign * (frac / float(1 << _BF8_MANT_BITS)) * (2.0 ** (1 - _BF8_EXP_BIAS)))
    if exp == _BF8_EXP_MAX:
        if frac == 0:
            return np.float32(sign * np.inf)
        return np.float32(np.nan)

    return np.float32(sign * (1.0 + frac / float(1 << _BF8_MANT_BITS)) * (2.0 ** (exp - _BF8_EXP_BIAS)))
