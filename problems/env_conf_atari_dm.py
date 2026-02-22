"""Consolidation module for Atari/DM-Control and CNN policy imports used by env_conf. Reduces env_conf fan-out.

Import this module before using atari: or dm: tags so handlers are registered with env_conf.
"""

import sys

from problems import env_conf

# Register with env_conf so get_env_conf can handle atari/dm tags without top-level import.
env_conf.register_atari_dm(sys.modules[__name__])


def get_cnn_mlp_policy_factory():
    from problems.cnn_mlp_policy import CNNMLPPolicyFactory

    return CNNMLPPolicyFactory


def get_atari_parsers_and_factories():
    from problems.atari_env import _parse_atari_tag
    from problems.cnn_mlp_policy import (
        AtariAgent57LiteFactory,
        AtariCNNPolicyFactory,
        AtariGaussianPolicyFactory,
    )

    return (
        _parse_atari_tag,
        AtariAgent57LiteFactory,
        AtariCNNPolicyFactory,
        AtariGaussianPolicyFactory,
    )


def get_dm_control_make():
    from problems.shimmy_dm_control import make as make_dm_control_env

    return make_dm_control_env


def get_atari_make():
    from problems.atari_env import make as make_atari_env

    return make_atari_env
