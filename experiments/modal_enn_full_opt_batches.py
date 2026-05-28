"""Full-optimization job keys, JSON payloads, and pending-job iteration for ENN Modal batches."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from analysis.fitting_time.fitting_time_enn_full_opt import (
    FULL_OPT_NUM_ARMS,
    FULL_OPT_NUM_DENOISE,
    FULL_OPT_POLICY_TAG,
    EnnFullOptTimingResult,
    enn_full_opt_checkpoint_ns,
    opt_name_for_index_driver,
)
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from analysis.sweep_plots_style import DEFAULT_SYNTH_10D_ENV_TAGS
from common.experiment_seeds import problem_seed_from_rep_index
from experiments import modal_enn_full_opt_batches_json as _full_opt_json
from experiments.enn_batch_job_params import (
    normalize_index_driver,
    validate_enn_batch_scalars,
)


def env_tag_slug(env_tag: str) -> str:
    return str(env_tag).replace(":", "_")


def full_opt_job_key(
    *,
    env_tag: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> str:
    drv = normalize_index_driver(index_driver).value
    slug = env_tag_slug(env_tag)
    return f"enn_full_opt_{slug}_pseed{int(problem_seed)}_nrep{int(num_reps)}_rep{int(rep_index)}_{drv}"


def full_opt_result_json_dest(
    output_dir: str | Path,
    *,
    env_tag: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
) -> Path:
    return Path(output_dir) / (
        f"{full_opt_job_key(env_tag=env_tag, problem_seed=problem_seed, rep_index=rep_index, num_reps=num_reps, index_driver=index_driver)}.json"
    )


def full_opt_result_to_payload(
    result: EnnFullOptTimingResult,
    *,
    num_reps: int,
) -> dict:
    return {
        "N": list(result.n),
        "proposal_elapsed_seconds": list(result.proposal_elapsed_seconds),
        "_meta": {
            "env_tag": result.env_tag,
            "opt_name": result.opt_name,
            "index_driver": result.index_driver.value,
            "policy_tag": FULL_OPT_POLICY_TAG,
            "problem_seed": int(result.problem_seed),
            "rep_index": int(result.rep_index),
            "num_reps": int(num_reps),
            "num_arms": FULL_OPT_NUM_ARMS,
            "num_denoise": FULL_OPT_NUM_DENOISE,
            "num_rounds": int(result.num_rounds),
            "stop_reason": str(result.stop_reason),
        },
    }


def iter_full_opt_jobs(
    output_dir: str | Path,
    index_driver: str,
    num_reps: int,
    *,
    iter_index_drivers: Callable[[str], tuple[EnnIncrementalIndexDriver, ...]],
    env_tags: tuple[str, ...] = DEFAULT_SYNTH_10D_ENV_TAGS,
) -> Iterable[tuple[str, tuple[str, int, int, int, str]]]:
    _, nr = validate_enn_batch_scalars(num_reps=num_reps, d=10)
    chk = enn_full_opt_checkpoint_ns()
    for env_tag in env_tags:
        for drv in iter_index_drivers(index_driver):
            opt_name = opt_name_for_index_driver(drv)
            for ri in range(nr):
                ps = problem_seed_from_rep_index(ri)
                dest = full_opt_result_json_dest(
                    output_dir,
                    env_tag=env_tag,
                    problem_seed=ps,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                )
                if _full_opt_json.full_opt_result_json_complete(
                    dest,
                    chk,
                    env_tag=env_tag,
                    problem_seed=ps,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                    opt_name=opt_name,
                ):
                    continue
                key = full_opt_job_key(
                    env_tag=env_tag,
                    problem_seed=ps,
                    rep_index=ri,
                    num_reps=nr,
                    index_driver=drv,
                )
                yield key, (env_tag, ps, ri, nr, drv.value)
