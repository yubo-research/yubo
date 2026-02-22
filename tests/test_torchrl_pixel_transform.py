"""Tests for PixelsToObservation transform."""

import torch
from tensordict import TensorDict

from rl.algos.backends.torchrl.common.pixel_transform import (
    AtariObservationTransform,
    PixelsToObservation,
    ensure_atari_obs_format,
)


def test_pixels_to_observation_call_root():
    """PixelsToObservation transforms root pixels to observation."""
    t = PixelsToObservation(size=84)
    # HWC uint8
    pixels = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    td = TensorDict({"pixels": pixels}, batch_size=[])
    out = t._call(td)
    obs = out["observation"]
    assert obs.shape == (3, 84, 84)
    assert obs.dtype == torch.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_pixels_to_observation_call_next():
    """PixelsToObservation transforms (next, pixels) to (next, observation)."""
    t = PixelsToObservation(size=84)
    pixels = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    td = TensorDict({("next", "pixels"): pixels}, batch_size=[])
    out = t._call(td)
    obs = out["next", "observation"]
    assert obs.shape == (3, 84, 84)
    assert obs.dtype == torch.float32


def test_pixels_to_observation_reset():
    """PixelsToObservation _reset transforms pixels in tensordict_reset."""
    t = PixelsToObservation(size=84)
    pixels = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    td_reset = TensorDict({"pixels": pixels}, batch_size=[])
    out = t._reset(None, td_reset)
    obs = out["observation"]
    assert obs.shape == (3, 84, 84)


def test_pixels_to_observation_transform_observation_spec():
    """transform_observation_spec adds observation and (next, observation)."""
    from torchrl.data import Composite, UnboundedContinuous

    t = PixelsToObservation(size=84)
    spec = Composite(
        pixels=UnboundedContinuous(shape=(84, 84, 3), dtype=torch.uint8),
        device="cpu",
    )
    out = t.transform_observation_spec(spec)
    assert "observation" in out.keys(True, True)
    assert ("next", "observation") in out.keys(True, True)
    assert out["observation"].shape == (3, 84, 84)
    assert out[("next", "observation")].shape == (3, 84, 84)


def test_pixels_to_observation_resize():
    """PixelsToObservation resizes non-84x84 input."""
    t = PixelsToObservation(size=64)
    pixels = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8)
    td = TensorDict({"pixels": pixels}, batch_size=[])
    out = t._call(td)
    obs = out["observation"]
    assert obs.shape == (3, 64, 64)


def test_ensure_atari_obs_format_hwc4_uint8():
    obs = torch.randint(0, 256, (84, 84, 4), dtype=torch.uint8)
    out = ensure_atari_obs_format(obs, size=84)
    assert out.shape == (4, 84, 84)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ensure_atari_obs_format_batch_hwc4_float255():
    obs = torch.rand(2, 84, 84, 4, dtype=torch.float32) * 255.0
    out = ensure_atari_obs_format(obs, size=84, scale_float_255=True)
    assert out.shape == (2, 4, 84, 84)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_atari_observation_transform_frame_stack_layout():
    t = AtariObservationTransform(size=84)
    stacked = torch.randint(0, 256, (4, 84, 84, 1), dtype=torch.uint8)
    td = TensorDict({"observation": stacked}, batch_size=[])
    out = t._call(td)
    obs = out["observation"]
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == torch.float32
