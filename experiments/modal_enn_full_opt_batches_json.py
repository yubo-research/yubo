"""Per-checkpoint full-optimization JSON completeness for ENN Modal batches."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments.enn_batch_job_params import normalize_index_driver

_FULL_OPT_META_REQUIRED: tuple[str, ...] = (
    "env_tag",
    "opt_name",
    "index_driver",
    "policy_tag",
    "problem_seed",
    "rep_index",
    "num_reps",
    "num_arms",
    "num_denoise",
    "num_rounds",
    "stop_reason",
)


def full_opt_meta_matches(
    meta: dict,
    *,
    env_tag: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    opt_name: str,
) -> bool:
    drv = normalize_index_driver(index_driver).value
    checks = (
        str(meta["env_tag"]) == str(env_tag),
        int(meta["problem_seed"]) == int(problem_seed),
        int(meta["rep_index"]) == int(rep_index),
        int(meta["num_reps"]) == int(num_reps),
        str(meta["index_driver"]).strip().lower() == drv,
        str(meta["opt_name"]) == str(opt_name),
        str(meta["policy_tag"]) == "pure-function",
        int(meta["num_arms"]) == 1,
        int(meta["num_denoise"]) == 1,
    )
    return all(checks)


def full_opt_result_json_complete(
    dest: str | Path,
    expected_n: tuple[int, ...],
    *,
    env_tag: str,
    problem_seed: int,
    rep_index: int,
    num_reps: int,
    index_driver: str | EnnIncrementalIndexDriver,
    opt_name: str,
) -> bool:
    path = Path(dest)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        meta = payload["_meta"]
        n_vals = tuple(int(n) for n in payload["N"])
        elapsed = payload["proposal_elapsed_seconds"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    tn = tuple(int(n) for n in expected_n)
    if n_vals != tn or len(elapsed) != len(tn):
        return False
    if not isinstance(meta, dict):
        return False
    if any(key not in meta for key in _FULL_OPT_META_REQUIRED):
        return False
    return full_opt_meta_matches(
        meta,
        env_tag=env_tag,
        problem_seed=problem_seed,
        rep_index=rep_index,
        num_reps=num_reps,
        index_driver=index_driver,
        opt_name=opt_name,
    )
