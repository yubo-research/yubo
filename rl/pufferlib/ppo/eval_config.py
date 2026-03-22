from __future__ import annotations

from problems.problem import build_problem
from rl.core.env_conf import build_seeded_env_conf_from_run, resolve_run_seeds
from rl.core.ppo_envs import _env_tag_for_problem_build, _maybe_register_atari_dm_backends


def resolve_eval_seeds(config) -> tuple[int, int]:
    resolved = resolve_run_seeds(seed=int(config.seed), problem_seed=config.problem_seed, noise_seed_0=config.noise_seed_0)
    return (int(resolved.problem_seed), int(resolved.noise_seed_0))


def build_eval_env_conf(config, *, obs_mode: str, is_atari_env_tag_fn, resolve_gym_env_name_fn):
    from problems.env_conf import EnvConf, GymConf

    tag = str(config.env_tag)
    _ = is_atari_env_tag_fn
    from_pixels = obs_mode == "pixels"

    def get_runtime_via_problem(
        env_tag: str,
        *,
        problem_seed: int,
        noise_seed_0: int,
        from_pixels: bool,
        pixels_only: bool,
    ):
        _maybe_register_atari_dm_backends(env_tag)
        adj = _env_tag_for_problem_build(env_tag, from_pixels=from_pixels)
        problem = build_problem(adj, "linear", problem_seed=int(problem_seed), noise_seed_0=int(noise_seed_0))
        env = problem.env
        if env.spec.env_name.startswith("dm_control/"):
            env.spec.pixels_only = bool(pixels_only)
        return env

    resolved = build_seeded_env_conf_from_run(
        env_tag=tag,
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=from_pixels,
        pixels_only=bool(config.pixels_only),
        get_env_conf_fn=get_runtime_via_problem,
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
