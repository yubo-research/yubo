#!/usr/bin/env python3
"""Batch UHD runs: local multiprocessing or Modal.

Usage:
  ./ops/uhd_batch.py local  <config.toml> --num-reps N [--workers W] [--results-dir DIR]
  ./ops/uhd_batch.py modal  <config.toml> --num-reps N [--results-dir DIR]
  ./ops/uhd_batch.py collect [--results-dir DIR]
  ./ops/uhd_batch.py status
  ./ops/uhd_batch.py cleanup
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

_HASH_EXCLUDE = frozenset({"problem_seed", "noise_seed_0"})
_DEFAULT_RESULTS = "results/uhd"
_APP_NAME = "yubo_uhd_batch"


def _config_hash(cfg: dict) -> str:
    d = {k: v for k, v in sorted(cfg.items()) if k not in _HASH_EXCLUDE}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _experiment_dir(results_dir: str, cfg: dict) -> Path:
    return Path(results_dir) / _config_hash(cfg)


def _trace_path(exp_dir: Path, i_rep: int) -> Path:
    return exp_dir / "traces" / f"{i_rep:05d}.jsonl"


def _gen_missing_reps(exp_dir: Path, num_reps: int, base_seed: int):
    for i in range(num_reps):
        tp = _trace_path(exp_dir, i)
        if tp.with_suffix(".done").exists():
            continue
        ps = base_seed + i
        yield i, ps, 10 * ps, tp


def _dict_to_toml(cfg: dict) -> str:
    lines = ["[uhd]"]
    for k, v in sorted(cfg.items()):
        if v is None:
            continue
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"{k} = {json.dumps(list(v))}")
    lines.append("")
    return "\n".join(lines)


def _parse_eval_lines(log_text: str) -> list[dict]:
    records = []
    for line in log_text.splitlines():
        s = line.strip()
        if not s.startswith("EVAL:"):
            continue
        parts = s[len("EVAL:") :].split()
        d: dict[str, str] = {}
        i = 0
        while i + 2 < len(parts):
            if parts[i + 1] == "=":
                d[parts[i]] = parts[i + 2]
                i += 3
            else:
                i += 1
        if "i_iter" in d and "mu" in d:
            records.append(
                {
                    "i_iter": int(d["i_iter"]),
                    "rreturn": float(d["mu"]),
                    "dt_prop": 0.0,
                    "dt_eval": 0.0,
                }
            )
    return records


def _run_subprocess(cfg: dict) -> tuple[str, int]:
    """Run exp_uhd.py local with a temp TOML. Returns (stdout, returncode)."""
    fd, tmp = tempfile.mkstemp(suffix=".toml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_dict_to_toml(cfg))
        result = subprocess.run(
            [sys.executable, "-u", "ops/exp_uhd.py", "local", tmp],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"SUBPROCESS ERROR:\n{result.stderr}", file=sys.stderr)
        return result.stdout, result.returncode
    finally:
        os.unlink(tmp)


def _write_trace(tp: Path, records: list[dict]) -> None:
    tp.parent.mkdir(parents=True, exist_ok=True)
    with open(tp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tp.with_suffix(".done").touch()


def _write_config(exp_dir: Path, cfg: dict) -> None:
    config = {k: v for k, v in cfg.items() if k not in _HASH_EXCLUDE}
    config["opt_name"] = config.get("optimizer", "mezo")
    config["num_arms"] = 1
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _load_toml(path: str) -> dict:
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("uhd", data)
    return {k.replace("-", "_"): v for k, v in section.items()}


# ---- Local batch ----


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


# ---- Modal batch ----

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
                f.write(_dict_to_toml(cfg))
            result = subprocess.run(
                [sys.executable, "-u", "/root/ops/exp_uhd.py", "local", tmp],
                capture_output=True,
                text=True,
                cwd="/root",
            )
            if result.returncode != 0:
                print(f"WORKER ERROR:\n{result.stderr}")
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


# ---- CLI ----


@click.group()
def _cli():
    pass


cli = _cli


@_cli.command(name="local")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--num-reps", type=int, required=True, help="Number of replications")
@click.option("--workers", type=int, default=1, help="Parallel workers")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def local_cmd(config_toml: str, num_reps: int, workers: int, results_dir: str) -> None:
    cfg = _load_toml(config_toml)
    _batch_local(cfg, num_reps, results_dir, workers)


@_cli.command(name="modal")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--num-reps", type=int, required=True, help="Number of replications")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def modal_cmd(config_toml: str, num_reps: int, results_dir: str) -> None:
    cfg = _load_toml(config_toml)
    _batch_modal(cfg, num_reps, results_dir)


@_cli.command(name="collect")
@click.option("--results-dir", type=str, default=_DEFAULT_RESULTS, help="Results directory")
def collect_cmd(results_dir: str) -> None:
    _collect(results_dir)


@_cli.command(name="status")
def status_cmd() -> None:
    _require_modal()
    rd = _results_dict()
    sd = _submitted_dict()
    click.echo(f"results_available = {rd.len()}")
    click.echo(f"submitted = {sd.len()}")


@_cli.command(name="cleanup")
def cleanup_cmd() -> None:
    _require_modal()
    for name in ["uhd_batch_results", "uhd_batch_submitted"]:
        try:
            modal.Dict.delete(name)
            click.echo(f"Deleted dict: {name}")
        except Exception as e:
            click.echo(f"Delete failed for {name}: {e!r}")


if __name__ == "__main__":
    cli()
