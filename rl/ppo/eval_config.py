from __future__ import annotations

from types import SimpleNamespace

from common.obs_mode import normalize_obs_mode
from rl.core.envs import conf_for_run, seeds
from rl.env_provider import get_env_conf_fn


def seed_pair(config) -> tuple[int, int]:
    out = seeds(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    return (int(out.problem_seed), int(out.noise_seed_0))


def eval_conf(config, *, obs_mode: str, resolve_gym_env_name_fn):
    tag = str(config.env_tag)
    mode = normalize_obs_mode(getattr(config, "obs_mode", obs_mode))
    out = conf_for_run(
        env_tag=tag,
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        obs_mode=mode,
        get_env_conf_fn=get_env_conf_fn(),
    )
    conf = out.env_conf
    if getattr(conf, "gym_conf", None) is not None:
        return conf
    env_name, env_kwargs = resolve_gym_env_name_fn(tag)
    conf.env_name = str(env_name)
    conf.problem_seed = int(out.problem_seed)
    conf.noise_seed_0 = int(out.noise_seed_0)
    conf.obs_mode = mode
    conf.kwargs = dict(env_kwargs)
    conf.gym_conf = SimpleNamespace(
        max_steps=1000,
        num_frames_skip=1,
        transform_state=False,
        state_space=getattr(getattr(conf, "gym_conf", None), "state_space", None),
    )
    return conf
