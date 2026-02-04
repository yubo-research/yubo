from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, NamedTuple

from .datum import Datum
from .opt_trajectories import collect_denoised_trajectory

_ROLLOUT_CTX: dict[str, Any] = {
    "env_tag": None,
    "problem_seed": None,
    "noise_seed_0": None,
    "env_conf": None,
    "policy": None,
}


class IterateResult(NamedTuple):
    data: list[Datum]
    dt_prop: float
    dt_eval: float


def _init_rollout_worker(env_tag, problem_seed, noise_seed_0, frozen_noise) -> None:
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf(
        env_tag,
        problem_seed=problem_seed,
        noise_level=None,
        noise_seed_0=noise_seed_0,
    )
    env_conf.frozen_noise = frozen_noise
    policy = default_policy(env_conf)
    _ROLLOUT_CTX.update(
        {
            "env_tag": env_tag,
            "problem_seed": problem_seed,
            "noise_seed_0": noise_seed_0,
            "env_conf": env_conf,
            "policy": policy,
        }
    )


def _rollout_from_params(args):
    (
        env_tag,
        problem_seed,
        noise_seed_0,
        frozen_noise,
        num_denoise,
        i_noise,
        flat_params,
    ) = args
    env_conf = _ROLLOUT_CTX.get("env_conf")
    policy = _ROLLOUT_CTX.get("policy")
    if (
        env_conf is None
        or policy is None
        or _ROLLOUT_CTX.get("env_tag") != env_tag
        or _ROLLOUT_CTX.get("problem_seed") != problem_seed
        or _ROLLOUT_CTX.get("noise_seed_0") != noise_seed_0
    ):
        _init_rollout_worker(env_tag, problem_seed, noise_seed_0, frozen_noise)
        env_conf = _ROLLOUT_CTX["env_conf"]
        policy = _ROLLOUT_CTX["policy"]
    else:
        env_conf.frozen_noise = frozen_noise
    policy.set_params(flat_params)
    traj, noise_seed = collect_denoised_trajectory(env_conf, policy, num_denoise, i_noise)
    return traj, noise_seed


def propose_policies(opt: Any, designer: Any, num_arms: int) -> tuple[list, float]:
    opt._telemetry.reset()
    t0 = time.time()
    policies = designer(opt._data, num_arms, telemetry=opt._telemetry)
    dt_prop = time.time() - t0
    return policies, float(dt_prop)


def _noise_delta(num_denoise: int | None) -> int:
    return 1 if num_denoise is None else int(num_denoise)


def _noise_indices(opt: Any, *, num_policies: int) -> list[int | None]:
    if opt._env_conf.frozen_noise:
        return [None for _ in range(num_policies)]
    delta = _noise_delta(opt._num_denoise)
    indices = [opt._i_noise + i * delta for i in range(num_policies)]
    opt._i_noise += delta * num_policies
    return indices


def _ensure_rollout_pool(opt: Any) -> None:
    if opt._rollout_pool is not None:
        return
    if opt._env_tag is None:
        raise ValueError("rollout_workers requires env_tag to reconstruct policies in workers")
    opt._rollout_pool = ProcessPoolExecutor(
        max_workers=opt._rollout_workers,
        initializer=_init_rollout_worker,
        initargs=(
            opt._env_tag,
            opt._env_conf.problem_seed,
            opt._env_conf.noise_seed_0,
            opt._env_conf.frozen_noise,
        ),
    )


def evaluate_policies(
    opt: Any,
    *,
    designer: Any,
    policies: list,
) -> tuple[list[Datum], float]:
    t0 = time.time()
    if opt._rollout_workers and opt._rollout_workers > 1:
        data = _evaluate_parallel(opt, designer=designer, policies=policies)
    else:
        data = _evaluate_sequential(opt, designer=designer, policies=policies)
    dt_eval = time.time() - t0
    return data, float(dt_eval)


def _evaluate_parallel(opt: Any, *, designer: Any, policies: list) -> list[Datum]:
    _ensure_rollout_pool(opt)
    noise_indices = _noise_indices(opt, num_policies=len(policies))
    flat_params = [p.get_params() for p in policies]
    args = [
        (
            opt._env_tag,
            opt._env_conf.problem_seed,
            opt._env_conf.noise_seed_0,
            opt._env_conf.frozen_noise,
            opt._num_denoise,
            noise_indices[i],
            flat_params[i],
        )
        for i in range(len(policies))
    ]
    results = list(opt._rollout_pool.map(_rollout_from_params, args))
    data = []
    for policy, (traj, _noise_seed) in zip(policies, results, strict=True):
        data.append(Datum(designer, policy, None, traj))
    return data


def _evaluate_sequential(opt: Any, *, designer: Any, policies: list) -> list[Datum]:
    data = []
    for policy in policies:
        if opt._env_conf.frozen_noise:
            i_noise = None
        else:
            i_noise = opt._i_noise
            opt._i_noise += _noise_delta(opt._num_denoise)
        traj, _noise_seed = collect_denoised_trajectory(opt._env_conf, policy, opt._num_denoise, i_noise)
        data.append(Datum(designer, policy, None, traj))
    return data


def iterate_step(opt: Any, designer: Any, num_arms: int) -> IterateResult:
    num_arms = int(num_arms)
    if num_arms <= 0:
        raise ValueError(num_arms)
    policies, dt_prop = propose_policies(opt, designer, num_arms)
    data, dt_eval = evaluate_policies(opt, designer=designer, policies=policies)
    return IterateResult(data=data, dt_prop=float(dt_prop), dt_eval=float(dt_eval))
