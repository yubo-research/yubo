from policies.actor_critic_mlp_policy import ActorCriticMLPPolicyFactory
from policies.mlp_policy import MLPPolicyFactory
from problems.bipedal_walker_policy import BipedalWalkerPolicy
from problems.env_conf_bindings import get_atari_dm_bindings
from problems.env_conf_constants import (
    _ATARI_DEFAULT_MAX_STEPS,
    _DM_CONTROL_DEFAULT_MAX_STEPS,
)
from problems.env_conf_policies import gaussian_policy_factory
from problems.env_conf_types import EnvConf, GymConf, _gym_conf
from problems.turbo_lunar_policy import TurboLunarPolicy


def _atari_pong_policy(env_conf):
    bindings = get_atari_dm_bindings()
    _env_id, policy_class = bindings.resolve_atari_from_tag("atari:Pong")
    return policy_class(env_conf)


# See https://paperswithcode.com/task/openai-gym
# num_frames_skip is not "frame_skip" in gymnasium. num_frames_skip is only used internally.
_gym_env_confs = {
    # 95
    "mcc": _gym_conf(
        "MountainCarContinuous-v0",
        gym_conf=GymConf(num_frames_skip=100),
    ),
    "pend": EnvConf("Pendulum-v1", gym_conf=GymConf(max_steps=200, num_frames_skip=100)),
    # 3580 - https://arxiv.org/pdf/1803.07055
    # 6600 - 2024 [??ref] k
    "ant": _gym_conf("Ant-v5"),
    "mpend": _gym_conf("InvertedPendulum-v5"),
    "macro": _gym_conf("InvertedDoublePendulum-v5"),
    # 325 - https://arxiv.org/pdf/1803.07055
    "swim": _gym_conf("Swimmer-v5"),
    "cheetah": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((32, 16)),
        rl_model={
            "ppo": {
                "backbone_hidden_sizes": (64, 64),
                "backbone_layer_norm": True,
                "share_backbone": True,
                "log_std_init": -0.5,
            },
            "sac": {
                "backbone_hidden_sizes": (256, 256),
                "backbone_activation": "relu",
                "backbone_layer_norm": False,
                "head_activation": "relu",
            },
        },
    ),
    "quadruped-run-64x64": _gym_conf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((64, 64)),
    ),
    "cheetah-16x16": _gym_conf(
        "HalfCheetah-v5",
        policy_class=MLPPolicyFactory((16, 16)),
    ),
    "cheetah-16x16-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=gaussian_policy_factory(variant="rl-gauss-tanh"),
    ),
    "cheetah-gauss": _gym_conf(
        "HalfCheetah-v5",
        policy_class=gaussian_policy_factory(variant="rl-gauss-small"),
    ),
    "reach": EnvConf("Reacher-v5", gym_conf=GymConf(max_steps=50)),
    # "push": EnvConf("Pusher-v4",  gym_conf=GymConf(max_steps=100)),
    "hop": _gym_conf("Hopper-v5"),
    "hop-gauss": _gym_conf(
        "Hopper-v5",
        policy_class=gaussian_policy_factory(variant="rl-gauss-small"),
    ),
    # 6900
    "human": _gym_conf("Humanoid-v5"),
    # 130,000 - https://arxiv.org/html/2304.12778
    "stand": _gym_conf("HumanoidStandup-v5"),
    "stand-mlp": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((32, 16)),
    ),
    "stand-mlp2": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((256, 128)),
    ),
    "stand-mlp3": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((1024, 600)),
    ),
    "stand-mlp4": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((4096, 2060)),
    ),
    "stand-mlp5": _gym_conf(
        "HumanoidStandup-v5",
        policy_class=MLPPolicyFactory((32000, 31000)),
    ),
    "bw": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
    ),
    "bw-linraw": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
            transform_state=False,
        ),
    ),
    # See https://github.com/hardmaru/estool/blob/b0954523e906d852287c6f515f34756c550ccf42/config.py#L309
    #  for config (i.e., (40,40))
    # https://arxiv.org/html/2304.12778 uses (16,)
    #
    "bw-mlp": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
        ),
        policy_class=MLPPolicyFactory((1024, 512, 256, 128)),
    ),
    "bw-heur": _gym_conf(
        "BipedalWalker-v3",
        gym_conf=GymConf(
            max_steps=1600,
            num_frames_skip=100,
            transform_state=False,
        ),
        policy_class=BipedalWalkerPolicy,
        noise_seed_0=1,
    ),
    # 300
    "lunar": _gym_conf(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
    ),
    # 300
    "lunar-mlp": _gym_conf(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
        policy_class=MLPPolicyFactory((16, 8)),
    ),
    "lunar-ac": _gym_conf(
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
        ),
        kwargs={"continuous": True},
        policy_class=ActorCriticMLPPolicyFactory((16, 8)),
    ),
    "tlunar": EnvConf(
        # TuRBO paper specifies v2, but that raises an exception now
        "LunarLander-v3",
        gym_conf=GymConf(
            max_steps=500,
            transform_state=False,
        ),
        kwargs={"continuous": False},
        policy_class=TurboLunarPolicy,
    ),
}


_dm_control_env_confs = {
    "dm_control/quadruped-run-v0": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((64, 64)),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
        rl_model={
            "ppo": {
                "backbone_hidden_sizes": (64, 64),
                "backbone_layer_norm": True,
                "share_backbone": True,
                "log_std_init": -0.5,
            },
            "sac": {
                "backbone_hidden_sizes": (256, 256),
                "backbone_activation": "relu",
                "backbone_layer_norm": True,
                "head_activation": "relu",
            },
        },
    ),
    "dm_control/quadruped-run-v0-small": EnvConf(
        "dm_control/quadruped-run-v0",
        policy_class=MLPPolicyFactory((4, 4)),
        max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
    ),
}


_atari_env_confs = {
    "atari-pong": EnvConf(
        "ALE/Pong-v5",
        policy_class=_atari_pong_policy,
        from_pixels=True,
        pixels_only=True,
        max_steps=_ATARI_DEFAULT_MAX_STEPS,
        rl_model={
            "ppo": {
                "backbone_name": "nature_cnn_atari",
                "backbone_hidden_sizes": (),
                "backbone_activation": "relu",
                "backbone_layer_norm": False,
                "actor_head_hidden_sizes": (512,),
                "critic_head_hidden_sizes": (512,),
                "head_activation": "relu",
                "share_backbone": True,
                "log_std_init": -0.5,
            }
        },
    ),
}
