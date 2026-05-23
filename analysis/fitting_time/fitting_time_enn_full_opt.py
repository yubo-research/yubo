"""Full TuRBO-ENN BO proposal-time snapshots at iteration-index checkpoints."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Sequence

from analysis.fitting_time.fitting_time_enn_incremental import (
    ENN_INCREMENTAL_CHECKPOINT_NS,
    EnnIncrementalIndexDriver,
)
from common.experiment_seeds import (
    global_seed_for_run,
    noise_seed_0_from_problem_seed,
    problem_seed_from_rep_index,
)
from common.seed_all import seed_all
from optimizer.optimizer import Optimizer
from problems.problem import build_problem

FULL_OPT_MAX_N = 100_000
FULL_OPT_NUM_ROUNDS = FULL_OPT_MAX_N
FULL_OPT_CHECKPOINT_NS: tuple[int, ...] = tuple(n for n in ENN_INCREMENTAL_CHECKPOINT_NS if n <= FULL_OPT_MAX_N)
FULL_OPT_NUM_ARMS = 1
FULL_OPT_NUM_DENOISE = 1
FULL_OPT_POLICY_TAG = "pure-function"
_OPT_NAME_FLAT = "turbo-enn-fit-ucb"
_OPT_NAME_HNSW = "turbo-enn-fit-ucb/idx=hnsw"


def enn_full_opt_checkpoint_ns() -> tuple[int, ...]:
    return FULL_OPT_CHECKPOINT_NS


@dataclass(frozen=True)
class EnnFullOptTimingResult:
    n: tuple[int, ...]
    proposal_elapsed_seconds: tuple[float, ...]
    env_tag: str
    opt_name: str
    index_driver: EnnIncrementalIndexDriver
    problem_seed: int
    rep_index: int
    num_rounds: int
    stop_reason: str


def opt_name_for_index_driver(index_driver: EnnIncrementalIndexDriver) -> str:
    if index_driver == EnnIncrementalIndexDriver.HNSW:
        return _OPT_NAME_HNSW
    return _OPT_NAME_FLAT


def _validate_problem_seed_for_rep_index(problem_seed: int, rep_index: int) -> int:
    ri = int(rep_index)
    ps = int(problem_seed)
    expected = problem_seed_from_rep_index(ri)
    if ps != expected:
        raise ValueError(f"problem_seed {ps} must equal 18 + rep_index (expected {expected} for rep_index {ri})")
    return ps


def _validate_q5_hyperparameters(*, num_arms: int, num_denoise: int, policy_tag: str) -> None:
    na = int(num_arms)
    if na != FULL_OPT_NUM_ARMS:
        raise ValueError(f"num_arms must be {FULL_OPT_NUM_ARMS}, got {na}")
    nd = int(num_denoise)
    if nd != FULL_OPT_NUM_DENOISE:
        raise ValueError(f"num_denoise must be {FULL_OPT_NUM_DENOISE}, got {nd}")
    pt = str(policy_tag)
    if pt != FULL_OPT_POLICY_TAG:
        raise ValueError(f"policy_tag must be {FULL_OPT_POLICY_TAG!r}, got {pt!r}")


def _install_wall_clock_stop_handler() -> None:
    try:
        signal.signal(signal.SIGTERM, _set_wall_clock_stop_requested)
    except (AttributeError, ValueError, OSError):
        pass


def _set_wall_clock_stop_requested(*_args) -> None:
    _wall_clock_stop_requested[0] = True


_wall_clock_stop_requested: list[bool] = [False]


def _finalize_stop_reason(
    *,
    next_idx: int,
    num_checkpoints: int,
    i_iter: int,
    num_rounds: int,
) -> str:
    if next_idx >= num_checkpoints:
        return "completed"
    if _wall_clock_stop_requested[0]:
        return "wall_clock_limit"
    if int(i_iter) >= int(num_rounds):
        return "num_rounds"
    return "interrupted"


def _validate_checkpoints(ckpts: tuple[int, ...]) -> None:
    if len(ckpts) == 0:
        raise ValueError("checkpoints must be non-empty")
    prev_n = 0
    for n_chk in ckpts:
        if int(n_chk) <= prev_n:
            raise ValueError(f"checkpoints must be strictly increasing, got {ckpts}")
        prev_n = int(n_chk)


def _validate_full_opt_checkpoints(ckpts: tuple[int, ...]) -> None:
    _validate_checkpoints(ckpts)
    for n_chk in ckpts:
        n_i = int(n_chk)
        if n_i > FULL_OPT_MAX_N:
            raise ValueError(f"full_optimization checkpoint N must be <= {FULL_OPT_MAX_N}, got {n_i}")


def _parse_full_opt_checkpoint_csv(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("checkpoints must be a comma-separated list of ints")
    try:
        ckpts = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("checkpoints must be a comma-separated list of ints") from exc
    _validate_full_opt_checkpoints(ckpts)
    return ckpts


def resolve_full_opt_checkpoints(raw: str | None) -> tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return enn_full_opt_checkpoint_ns()
    return _parse_full_opt_checkpoint_csv(str(raw))


def _snapshots_from_iter_counts(
    i_iter: int,
    cum_dt_proposing: float,
    ckpts: tuple[int, ...],
    next_idx: int,
    ns: list[int],
    elapsed: list[float],
) -> int:
    idx = next_idx
    while idx < len(ckpts) and i_iter == int(ckpts[idx]):
        ns.append(int(ckpts[idx]))
        elapsed.append(float(cum_dt_proposing))
        idx += 1
    return idx


def benchmark_enn_full_optimization_proposal_timing(
    *,
    env_tag: str,
    problem_seed: int,
    rep_index: int,
    index_driver: EnnIncrementalIndexDriver = EnnIncrementalIndexDriver.FLAT,
    checkpoints: Sequence[int] | None = None,
    num_rounds: int = FULL_OPT_NUM_ROUNDS,
    num_arms: int = FULL_OPT_NUM_ARMS,
    num_denoise: int = FULL_OPT_NUM_DENOISE,
    policy_tag: str = FULL_OPT_POLICY_TAG,
) -> EnnFullOptTimingResult:
    from common.collector import Collector

    ckpts = tuple(checkpoints) if checkpoints is not None else enn_full_opt_checkpoint_ns()
    _validate_full_opt_checkpoints(ckpts)
    ri = int(rep_index)
    ps = _validate_problem_seed_for_rep_index(problem_seed, ri)
    nr = int(num_rounds)
    if nr < 1:
        raise ValueError(f"num_rounds must be >= 1, got {nr}")
    _validate_q5_hyperparameters(
        num_arms=num_arms,
        num_denoise=num_denoise,
        policy_tag=policy_tag,
    )

    opt_name = opt_name_for_index_driver(index_driver)
    _wall_clock_stop_requested[0] = False
    _install_wall_clock_stop_handler()

    ns: list[int] = []
    proposal_elapsed: list[float] = []
    next_idx = 0
    stop_reason = "completed"
    opt = None

    try:
        problem = build_problem(
            env_tag,
            policy_tag,
            problem_seed=ps,
            noise_seed_0=noise_seed_0_from_problem_seed(ps),
        )
        env_conf = problem.env
        seed_all(global_seed_for_run(ps))
        policy = problem.build_policy()

        opt = Optimizer(
            Collector(),
            env_conf=env_conf,
            policy_tag=policy_tag,
            policy=policy,
            num_arms=int(num_arms),
            num_denoise_measurement=int(num_denoise),
            num_denoise_passive=None,
            opt_name=opt_name,
            num_rounds=nr,
        )
        opt.initialize(opt_name)

        for _ in range(nr):
            if _wall_clock_stop_requested[0]:
                break
            opt.iterate()
            next_idx = _snapshots_from_iter_counts(
                int(opt._i_iter),
                float(opt._cum_dt_proposing),
                ckpts,
                next_idx,
                ns,
                proposal_elapsed,
            )
            if next_idx >= len(ckpts):
                break
    finally:
        if opt is not None:
            opt.stop()
        stop_reason = _finalize_stop_reason(
            next_idx=next_idx,
            num_checkpoints=len(ckpts),
            i_iter=int(opt._i_iter) if opt is not None else 0,
            num_rounds=nr,
        )

    return EnnFullOptTimingResult(
        n=tuple(ns),
        proposal_elapsed_seconds=tuple(proposal_elapsed),
        env_tag=str(env_tag),
        opt_name=opt_name,
        index_driver=index_driver,
        problem_seed=ps,
        rep_index=ri,
        num_rounds=nr,
        stop_reason=stop_reason,
    )


def collect_full_opt_snapshots_from_optimizer(
    opt,
    *,
    checkpoints: Sequence[int],
    max_iterations: int,
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    ckpts = tuple(int(n) for n in checkpoints)
    _validate_full_opt_checkpoints(ckpts)
    _wall_clock_stop_requested[0] = False
    nr = int(max_iterations)
    if nr < 1:
        raise ValueError(f"max_iterations must be >= 1, got {nr}")

    ns: list[int] = []
    proposal_elapsed: list[float] = []
    next_idx = 0
    stop_reason = "completed"

    for _ in range(nr):
        opt.iterate()
        next_idx = _snapshots_from_iter_counts(
            int(opt._i_iter),
            float(opt._cum_dt_proposing),
            ckpts,
            next_idx,
            ns,
            proposal_elapsed,
        )
        if next_idx >= len(ckpts):
            break

    if hasattr(opt, "stop"):
        opt.stop()

    i_done = int(opt._i_iter)
    stop_reason = _finalize_stop_reason(
        next_idx=next_idx,
        num_checkpoints=len(ckpts),
        i_iter=i_done,
        num_rounds=nr,
    )

    return tuple(ns), tuple(proposal_elapsed), stop_reason
