#!/usr/bin/env python

from __future__ import annotations

import contextlib
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any


def worker(cmd: str) -> int:
    return os.system(cmd)


def _normalize_config_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "to_dict") and callable(getattr(raw, "to_dict")):
        return dict(raw.to_dict())
    raise TypeError(f"Unsupported config type: {type(raw).__name__}. Expected dict or ExperimentConfig-like object.")


def _run_one_config(config_dict: dict[str, Any], log_path: str) -> None:
    from experiments.experiment_sampler import ExperimentConfig, sampler, scan_local

    config = ExperimentConfig.from_dict(config_dict)
    with open(log_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            sampler(config, distributor_fn=scan_local)


def run_batch(configs, b_dry_run):
    processes = []

    for config_like in configs:
        config_dict = _normalize_config_dict(config_like)
        exp_dir = str(config_dict["exp_dir"])
        opt_name = str(config_dict["opt_name"])
        logs_dir = Path(exp_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(logs_dir / opt_name)

        env_tag = config_dict.get("env_tag", "?")
        print(f"RUN: env_tag={env_tag} opt_name={opt_name} log={log_path}")

        if not b_dry_run:
            process = multiprocessing.Process(target=_run_one_config, args=(config_dict, log_path))
            processes.append(process)
            process.start()

    if not b_dry_run:
        for process in processes:
            process.join()
    print("DONE_BATCH")


def run(configs, max_parallel, b_dry_run=False):
    from experiments.batch_util import run_in_batches

    run_in_batches(configs, max_parallel, run_batch, b_dry_run=b_dry_run, num_threads=16)


def prep_d_argss(batch_tag: str, *, results_dir: str = "results"):
    import experiments.batch_preps as batch_preps

    preps = {k: v for k, v in batch_preps.__dict__.items() if k.startswith("prep_") and callable(v)}

    fn = preps.get(batch_tag)
    if fn is None and not batch_tag.startswith("prep_"):
        fn = preps.get(f"prep_{batch_tag}")

    assert fn is not None, f"Unknown batch_tag: {batch_tag} (known: {sorted(preps.keys())})"
    return fn(results_dir)


def run_from_batch_tag(batch_tag: str, *, max_parallel: int = 5, dry_run: bool = False, results_dir: str = "results") -> None:
    configs = prep_d_argss(batch_tag, results_dir=results_dir)
    t_0 = time.time()
    run(configs, max_parallel=max_parallel, b_dry_run=dry_run)
    t_f = time.time()
    print(f"TIME_ALL: {t_f - t_0:.2f}")
    print("DONE_ALL")
