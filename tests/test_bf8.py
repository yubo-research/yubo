from __future__ import annotations

import numpy as np


def test_bf8_roundtrip_basic_values():
    from common.bf8 import bf8_decode, bf8_encode

    x = np.asarray([0.0, -0.0, 1.0, -2.0, 0.5, 3.5], dtype=np.float32)
    enc = bf8_encode(x)
    dec = bf8_decode(enc)

    np.testing.assert_allclose(dec[:4], np.asarray([0.0, -0.0, 1.0, -2.0], dtype=np.float32))
    assert dec[4] > 0.0
    assert dec[5] > 0.0


def test_bf8_handles_special_values():
    from common.bf8 import bf8_decode, bf8_encode

    x = np.asarray([np.inf, -np.inf, np.nan], dtype=np.float32)
    enc = bf8_encode(x)
    dec = bf8_decode(enc)

    assert np.isinf(dec[0]) and dec[0] > 0
    assert np.isinf(dec[1]) and dec[1] < 0
    assert np.isnan(dec[2])


def test_bf8_precision_is_limited():
    from common.bf8 import bf8_decode, bf8_encode

    x = np.asarray([1.125, 1.25, 1.375], dtype=np.float32)
    dec = bf8_decode(bf8_encode(x))

    assert np.unique(dec).size <= 3


def test_bf8_tree_roundtrip_nested_structures():
    from common.bf8 import bf8_decode_tree, bf8_encode_tree

    tree = {
        "a": np.asarray([0.0, 1.5], dtype=np.float32),
        "b": [
            np.asarray([-2.0], dtype=np.float32),
            (np.asarray([3.0], dtype=np.float32), None),
        ],
    }

    encoded = bf8_encode_tree(tree)
    decoded = bf8_decode_tree(encoded)

    np.testing.assert_allclose(decoded["a"], np.asarray([0.0, 1.5], dtype=np.float32), atol=0.5)
    np.testing.assert_allclose(decoded["b"][0], np.asarray([-2.0], dtype=np.float32), atol=0.5)
    np.testing.assert_allclose(decoded["b"][1][0], np.asarray([3.0], dtype=np.float32), atol=0.5)
    assert decoded["b"][1][1] is None


def test_bf8_tree_roundtrip_preserves_torch_tensors():
    import torch

    from common.bf8 import bf8_decode_tree, bf8_encode_tree

    x = torch.tensor([1.0, -2.0], dtype=torch.float32)
    decoded = bf8_decode_tree(bf8_encode_tree({"x": x}))["x"]

    assert isinstance(decoded, torch.Tensor)
    assert decoded.dtype == x.dtype
    assert decoded.device == x.device
    torch.testing.assert_close(decoded, x)
