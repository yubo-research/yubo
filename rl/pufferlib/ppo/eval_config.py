from __future__ import annotations

from rl.core.env_conf import build_seeded_env_conf_from_run, resolve_run_seeds


def resolve_eval_seeds(config) -> tuple[int, int]:
    resolved = resolve_run_seeds(
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
    )
    return (int(resolved.problem_seed), int(resolved.noise_seed_0))


def build_eval_env_conf(config, *, obs_mode: str, is_atari_env_tag_fn, resolve_gym_env_name_fn):
    from problems.env_conf import EnvConf, GymConf, get_env_conf

    tag = str(config.env_tag)
    _ = is_atari_env_tag_fn
    from_pixels = obs_mode == "pixels"
    resolved = build_seeded_env_conf_from_run(
        env_tag=tag,
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=from_pixels,
        pixels_only=bool(config.pixels_only),
        get_env_conf_fn=get_env_conf,
    )
    env_conf = resolved.env_conf
    problem_seed = int(resolved.problem_seed)
    noise_seed_0 = int(resolved.noise_seed_0)
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
