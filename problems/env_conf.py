import copy

from policies.mlp_policy import MLPPolicyFactory
from problems.env_conf_bindings import (
    get_atari_dm_bindings,
    register_atari_dm_bindings_loader,
)
from problems.env_conf_constants import (
    _ATARI_DEFAULT_MAX_STEPS,
    _DM_CONTROL_DEFAULT_MAX_STEPS,
)
from problems.env_conf_parse import parse_tag_options
from problems.env_conf_presets import (
    _atari_env_confs,
    _dm_control_env_confs,
    _gym_env_confs,
)
from problems.env_conf_rl import resolve_rl_model_defaults
from problems.env_conf_types import EnvConf, GymConf
from problems.environment_spec import needs_atari_dm_bindings
from problems.isaaclab_env_adapters import DEFAULT_ISAACLAB_MAX_STEPS, is_isaaclab_env_tag
from problems.linear_policy import LinearPolicy
from problems.pure_function_policy import PureFunctionPolicy


__all__ = [
    "EnvConf",
    "GymConf",
    "default_policy",
    "get_env_conf",
    "needs_atari_dm_bindings",
    "register_atari_dm_bindings_loader",
    "resolve_rl_model_defaults",
    "_atari_env_confs",
    "_dm_control_env_confs",
    "_gym_env_confs",
]


def _ac_mlp_policy_factory(
    hidden_sizes: tuple[int, ...],
    *,
    share_backbone: bool = True,
    log_std_init: float = 0.0,
):
    _ns: dict = {}
    exec("from policies.actor_critic_mlp_policy import ActorCriticMLPPolicyFactory", _ns)  # noqa: S102
    ActorCriticMLPPolicyFactory = _ns["ActorCriticMLPPolicyFactory"]
    return ActorCriticMLPPolicyFactory(hidden_sizes, share_backbone=share_backbone, log_std_init=log_std_init)


def get_env_conf(
    tag,
    problem_seed=None,
    noise_level=None,
    noise_seed_0=None,
    from_pixels=None,
    pixels_only=None,
    atari_preprocess=None,
):
    tag, frozen_noise, from_pixels = parse_tag_options(tag, from_pixels)
    pix_only = pixels_only if pixels_only is not None else True

    if tag in _gym_env_confs:
        ec = copy.deepcopy(_gym_env_confs[tag])
    elif tag in _dm_control_env_confs:
        ec = copy.deepcopy(_dm_control_env_confs[tag])
    elif tag in _atari_env_confs:
        ec = copy.deepcopy(_atari_env_confs[tag])
    elif tag.startswith("dm:") or tag.startswith("dm_control/"):
        use_pixels = from_pixels if from_pixels is not None else False
        bindings = get_atari_dm_bindings()
        env_name, policy_cls = bindings.resolve_dm_control_from_tag(tag, bool(use_pixels))
        ec = EnvConf(
            env_name,
            policy_class=policy_cls,
            from_pixels=use_pixels,
            pixels_only=pix_only,
            max_steps=_DM_CONTROL_DEFAULT_MAX_STEPS,
        )
    elif tag.startswith("atari:") or tag.startswith("ALE/"):
        bindings = get_atari_dm_bindings()
        env_id, policy_cls = bindings.resolve_atari_from_tag(tag)
        ec = EnvConf(
            env_id,
            policy_class=policy_cls,
            from_pixels=True,
            pixels_only=True,
            max_steps=_ATARI_DEFAULT_MAX_STEPS,
        )
    elif is_isaaclab_env_tag(tag):
        ec = EnvConf(
            tag,
            policy_class=MLPPolicyFactory((64, 64)),
            gym_conf=GymConf(max_steps=DEFAULT_ISAACLAB_MAX_STEPS, num_frames_skip=1, transform_state=False),
            max_steps=DEFAULT_ISAACLAB_MAX_STEPS,
            problem_seed=problem_seed,
            noise_level=noise_level,
            noise_seed_0=noise_seed_0,
            frozen_noise=frozen_noise,
            rl_model={
                "ppo": {
                    "backbone_name": "mlp",
                    "backbone_hidden_sizes": (64, 64),
                    "backbone_activation": "silu",
                    "backbone_layer_norm": True,
                    "actor_head_hidden_sizes": (),
                    "critic_head_hidden_sizes": (),
                    "head_activation": "silu",
                    "share_backbone": True,
                    "log_std_init": -0.5,
                },
                "sac": {
                    "backbone_name": "mlp",
                    "backbone_hidden_sizes": (64, 64),
                    "backbone_activation": "silu",
                    "backbone_layer_norm": True,
                    "actor_head_hidden_sizes": (),
                    "critic_head_hidden_sizes": (),
                    "head_activation": "silu",
                },
            },
        )
    else:
        ec = EnvConf(
            tag,
            problem_seed=problem_seed,
            noise_level=noise_level,
            noise_seed_0=noise_seed_0,
            frozen_noise=frozen_noise,
        )

    ec.problem_seed = problem_seed
    ec.noise_seed_0 = noise_seed_0
    ec.frozen_noise = frozen_noise
    ec.env_tag = tag
    if atari_preprocess is not None:
        if not isinstance(atari_preprocess, dict):
            raise TypeError("atari_preprocess must be a dict when provided.")
        if not str(ec.env_name).startswith("ALE/"):
            raise ValueError("atari_preprocess is only valid for Atari envs (ALE/*).")
        ec.atari_preprocess = copy.deepcopy(atari_preprocess)
    return ec


def default_policy(env_conf):
    has_policy_or_gym = env_conf.policy_class is not None or env_conf.gym_conf is not None
    missing_spaces = getattr(env_conf, "state_space", None) is None or getattr(env_conf, "action_space", None) is None
    if has_policy_or_gym and missing_spaces:
        env_conf.ensure_spaces()
    if env_conf.policy_class is not None:
        return env_conf.policy_class(env_conf)
    elif env_conf.gym_conf is not None:
        return LinearPolicy(env_conf)
    else:  # env_conf.env_name[:2] == "f:":
        return PureFunctionPolicy(env_conf)
