import pytest
import torch

from problems.pixel_atari_spatial import atari_in_channels_from_obs_shape, atari_obs_to_nchw


def test_atari_in_channels_from_obs_shape_nhwc_84():
    assert atari_in_channels_from_obs_shape((4, 84, 84), require_spatial_84=True) == 4
    assert atari_in_channels_from_obs_shape((4, 84, 84), require_spatial_84=False) == 4
    with pytest.raises(AssertionError, match="84x84"):
        atari_in_channels_from_obs_shape((4, 1, 1), require_spatial_84=True)


def test_atari_obs_to_nchw_hwc_stacks_to_nchw():
    x = torch.zeros(1, 84, 84, 4)
    y = atari_obs_to_nchw(x)
    assert y.shape == (1, 4, 84, 84)


def test_atari_obs_to_nchw_already_nchw():
    x = torch.zeros(2, 4, 84, 84)
    y = atari_obs_to_nchw(x)
    assert y.shape == (2, 4, 84, 84)


def test_atari_obs_to_nchw_4d_trailing_one_hwc_block():
    x = torch.zeros(1, 84, 84, 1)
    y = atari_obs_to_nchw(x)
    assert y.shape[0] == 1 and y.shape[1] == 1
    assert y.shape[2] == 84 and y.shape[3] == 84
