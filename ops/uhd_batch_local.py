import sys
from pathlib import Path

import click

from ops.uhd_batch_core import (
    _gen_missing_reps,
    _parse_eval_lines,
    _run_subprocess,
    _write_config,
    _write_trace,
)


def _local_worker(args):
    cfg, i_rep, problem_seed, noise_seed_0, tp_str = args
    tp = Path(tp_str)
    print(f"REP {i_rep}: problem_seed={problem_seed}", file=sys.stderr)
    job_cfg = dict(cfg)
    job_cfg["problem_seed"] = problem_seed
    job_cfg["noise_seed_0"] = noise_seed_0
    stdout, rc = _run_subprocess(job_cfg)
    records = _parse_eval_lines(stdout)
    if not records:
        print(f"FAILED REP {i_rep}: exit={rc}, no EVAL records", file=sys.stderr)
        return
    _write_trace(tp, records)
    print(f"DONE REP {i_rep}: {len(records)} records", file=sys.stderr)


def _batch_local(cfg: dict, num_reps: int, results_dir: str, num_workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from ops.uhd_batch_core import _experiment_dir

    base_seed = int(cfg.get("problem_seed", 18))
    exp_dir = _experiment_dir(results_dir, cfg)
    _write_config(exp_dir, cfg)
    jobs = [(cfg, i_rep, ps, ns, str(tp)) for i_rep, ps, ns, tp in _gen_missing_reps(exp_dir, num_reps, base_seed)]
    if not jobs:
        click.echo(f"All {num_reps} reps done in {exp_dir}")
        return

    click.echo(f"Running {len(jobs)} missing reps (of {num_reps}) with {num_workers} workers")
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(_local_worker, jobs))
    click.echo(f"BATCH DONE: {len(jobs)} reps -> {exp_dir}")
