import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from ops.uhd_batch_core import (
    _APP_NAME,
    _config_hash,
    _experiment_dir,
    _gen_missing_reps,
    _parse_eval_lines,
    _trace_path,
    _write_config,
    _write_trace,
)


try:
    import modal

    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False


if _HAS_MODAL:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _MODAL_DIRS = ("ops", "optimizer", "problems", "common", "sampling", "embedding")
    _batch_image = (
        modal.Image.debian_slim(python_version="3.11.9")
        .apt_install("swig")
        .pip_install(
            "torch==2.3.1",
            "torchvision==0.18.1",
            "numpy==1.26.4",
            "gymnasium==1.2.0",
            "gymnasium[mujoco]",
            "gymnasium[box2d]",
            "mujoco==3.3.3",
            "scipy==1.15.3",
            "click==8.3.1",
        )
        .env({"PYTHONPATH": "/root"})
    )
    for _d in _MODAL_DIRS:
        _batch_image = _batch_image.add_local_dir(str(_PROJECT_ROOT / _d), remote_path=f"/root/{_d}")

    batch_app = modal.App(name=_APP_NAME)
    app = batch_app

    def _results_dict():
        return modal.Dict.from_name("uhd_batch_results", create_if_missing=True)

    def _submitted_dict():
        return modal.Dict.from_name("uhd_batch_submitted", create_if_missing=True)

    @batch_app.function(
        image=_batch_image,
        max_containers=200,
        timeout=4 * 60 * 60,
    )
    def uhd_batch_worker(job):
        key, cfg = job
        print(f"WORKER: key={key}")
        fd, tmp = tempfile.mkstemp(suffix=".toml")
        try:
            with os.fdopen(fd, "w") as f:
                from ops.uhd_batch_core import _dict_to_toml

                f.write(_dict_to_toml(cfg))
            result = subprocess.run(
                [sys.executable, "-u", "/root/ops/exp_uhd.py", "local", tmp],
                capture_output=True,
                text=True,
                cwd="/root",
            )
            if result.returncode != 0:
                raise RuntimeError(f"Subprocess failed with exit {result.returncode}:\n{result.stderr}")
            _results_dict()[key] = result.stdout
        finally:
            os.unlink(tmp)

    @batch_app.function(
        image=_batch_image,
        max_containers=20,
        timeout=60 * 60,
    )
    def uhd_batch_resubmitter(batch_of_jobs):
        submitted = _submitted_dict()
        todo = []
        for key, cfg in batch_of_jobs:
            if key in submitted:
                continue
            submitted[key] = True
            todo.append((key, cfg))
        print(f"RESUBMITTER: {len(todo)} new jobs")
        worker_fn = modal.Function.from_name(_APP_NAME, "uhd_batch_worker")
        worker_fn.spawn_map(todo)

    @batch_app.function(image=_batch_image, max_containers=1, timeout=60 * 60)
    def uhd_batch_deleter(keys):
        rd = _results_dict()
        for key in keys:
            try:
                del rd[key]
            except KeyError:
                pass

else:
    batch_app = None
    app = None
    uhd_batch_worker = None
    uhd_batch_resubmitter = None
    uhd_batch_deleter = None

    def _results_dict():
        _require_modal()
        raise AssertionError("unreachable")

    def _submitted_dict():
        _require_modal()
        raise AssertionError("unreachable")


def _require_modal():
    if not _HAS_MODAL:
        raise click.ClickException("modal is not installed; run: pip install modal")


def _batch_modal(cfg: dict, num_reps: int, results_dir: str) -> None:
    _require_modal()
    base_seed = int(cfg.get("problem_seed", 18))
    exp_dir = _experiment_dir(results_dir, cfg)
    cfg_hash = _config_hash(cfg)
    _write_config(exp_dir, cfg)

    batch: list[tuple[str, dict]] = []
    n_submitted = 0

    def _flush():
        nonlocal batch
        fn = modal.Function.from_name(_APP_NAME, "uhd_batch_resubmitter")
        fn.spawn(batch)
        batch = []

    for i_rep, ps, ns, _tp in _gen_missing_reps(exp_dir, num_reps, base_seed):
        key = f"{cfg_hash}-{i_rep:05d}"
        job_cfg = dict(cfg)
        job_cfg["problem_seed"] = ps
        job_cfg["noise_seed_0"] = ns
        batch.append((key, job_cfg))
        n_submitted += 1
        if len(batch) >= 200:
            _flush()
    if batch:
        _flush()

    click.echo(f"Submitted {n_submitted} reps for {cfg_hash}")


def _collect(results_dir: str) -> None:
    _require_modal()
    rd = _results_dict()
    click.echo(f"Results available: {rd.len()}")

    collected: set[str] = set()
    for key, log_text in rd.items():
        if not isinstance(log_text, str):
            continue
        parts = key.rsplit("-", 1)
        if len(parts) != 2:
            continue
        cfg_hash, rep_str = parts
        try:
            i_rep = int(rep_str)
        except ValueError:
            continue

        tp = _trace_path(Path(results_dir) / cfg_hash, i_rep)
        if not tp.with_suffix(".done").exists():
            records = _parse_eval_lines(log_text)
            if records:
                _write_trace(tp, records)
                click.echo(f"Collected: {key} ({len(records)} records)")
            else:
                click.echo(f"Warning: {key} has no records")
        collected.add(key)

    if collected:
        fn = modal.Function.from_name(_APP_NAME, "uhd_batch_deleter")
        fn.spawn(list(collected))
    click.echo(f"Collected {len(collected)} results")
