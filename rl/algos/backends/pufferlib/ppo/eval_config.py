"""Eval environment construction helpers for pufferlib PPO."""

from __future__ import annotations


def resolve_eval_seeds(config) -> tuple[int, int]:
    from rl.algos.seed_util import resolve_noise_seed_0, resolve_problem_seed

    problem_seed = resolve_problem_seed(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
    )
    noise_seed_0 = resolve_noise_seed_0(
        problem_seed=problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    return int(problem_seed), int(noise_seed_0)


def build_eval_env_conf(
    config,
    *,
    obs_mode: str,
    is_atari_env_tag_fn,
    resolve_gym_env_name_fn,
):
    from problems.env_conf import EnvConf, GymConf, get_env_conf

    tag = str(config.env_tag)
    if is_atari_env_tag_fn(tag) or tag.startswith("dm:") or tag.startswith("dm_control/"):
        import problems.env_conf_atari_dm  # noqa: F401

    problem_seed, noise_seed_0 = resolve_eval_seeds(config)
    from_pixels = obs_mode == "pixels"

    env_conf = get_env_conf(
        tag,
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        from_pixels=from_pixels,
        pixels_only=bool(config.pixels_only),
    )
    if getattr(env_conf, "gym_conf", None) is not None:
        return env_conf

    env_name, env_kwargs = resolve_gym_env_name_fn(tag)
    return EnvConf(
        env_name,
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        from_pixels=from_pixels,
        pixels_only=bool(config.pixels_only),
        gym_conf=GymConf(max_steps=1000, num_frames_skip=1, transform_state=False),
        kwargs=env_kwargs,
    )
