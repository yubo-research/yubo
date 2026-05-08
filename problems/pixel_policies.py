"""CNN + MLP policy for pixel-based observations (e.g. dm_control from_pixels)."""

from __future__ import annotations

from problems.pixel_policies_atari_agent57 import AtariAgent57LiteFactory, AtariAgent57LitePolicy
from problems.pixel_policies_atari_cnn_gauss import AtariCNNPolicy, AtariCNNPolicyFactory
from problems.pixel_policies_atari_gaussian import AtariGaussianPolicy, AtariGaussianPolicyFactory
from problems.pixel_policies_cnn_mlp import CNNMLPPolicy, CNNMLPPolicyFactory
from problems.pixel_policies_encoders import init_linear_and_conv, nature_cnn_encoder, obs_space_from_env_conf, tiny_atari_cnn_encoder

__all__ = [
    "AtariAgent57LiteFactory",
    "AtariAgent57LitePolicy",
    "AtariCNNPolicy",
    "AtariCNNPolicyFactory",
    "AtariGaussianPolicy",
    "AtariGaussianPolicyFactory",
    "CNNMLPPolicy",
    "CNNMLPPolicyFactory",
    "init_linear_and_conv",
    "nature_cnn_encoder",
    "obs_space_from_env_conf",
    "tiny_atari_cnn_encoder",
]
