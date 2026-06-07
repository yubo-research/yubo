import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ops.modal_cli_common import stop_app_and_delete_dicts  # noqa: E402
from ops.modal_enn_image import add_enn_to_image, enn_project_root  # noqa: E402
from ops.uhd_batch_core import (  # noqa: E402
    _APP_NAME,
    _config_hash,
    _experiment_dir,
    _gen_missing_reps,
    _parse_eval_lines,
    _trace_path,
    _write_config,
    _write_trace,
)

_UHD_BATCH_DICTS = ("uhd_batch_results", "uhd_batch_submitted")

try:
    import modal

    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False


if _HAS_MODAL:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _MODAL_DIRS = (
        "ops",
        "optimizer",
        "problems",
        "common",
        "sampling",
        "embedding",
        "policies",
        "rl",
    )
    _ENN_ROOT = enn_project_root(_PROJECT_ROOT)
    _batch_image = (
        modal.Image.debian_slim(python_version="3.11.9")
        .apt_install(
            "swig",
            "curl",
            "build-essential",
            "libopenblas-dev",
            "patchelf",
            "cmake",
            "ninja-build",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            'echo "export PATH=$HOME/.cargo/bin:$PATH" >> ~/.bashrc',
        )
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
            "maturin>=1.0",
        )
        .env({"PYTHONPATH": "/root"})
    )
    for _d in _MODAL_DIRS:
        _batch_image = _batch_image.add_local_dir(
            str(_PROJECT_ROOT / _d),
            remote_path=f"/root/{_d}",
            copy=True,
        )
    _batch_image = add_enn_to_image(_batch_image, _ENN_ROOT)

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


def _deploy_uhd_batch_app() -> None:
    root = Path(__file__).resolve().parents[1]
    click.echo("Deploying Modal app yubo_uhd_batch...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        ["modal", "deploy", "ops/uhd_batch_modal.py"],
        cwd=root,
        env=env,
    )
    if result.returncode != 0:
        raise click.ClickException("modal deploy ops/uhd_batch_modal.py failed; deploy manually then retry")


def _uhd_batch_app_ready() -> bool:
    try:
        modal.Function.from_name(_APP_NAME, "uhd_batch_resubmitter")
    except modal.exception.NotFoundError:
        return False
    return True


def _ensure_uhd_batch_app(*, deploy: bool = True) -> None:
    _require_modal()
    if _uhd_batch_app_ready():
        return
    if not deploy:
        raise click.ClickException("Modal app yubo_uhd_batch is not deployed. Run: ./ops/uhd_batch.py deploy")
    _deploy_uhd_batch_app()
    if not _uhd_batch_app_ready():
        raise click.ClickException("Modal app yubo_uhd_batch still not found after deploy")


def _stop_uhd_batch() -> None:
    _require_modal()
    stop_app_and_delete_dicts(app_name=_APP_NAME, dict_names=_UHD_BATCH_DICTS)


def _batch_modal(
    cfg: dict,
    num_reps: int,
    results_dir: str,
    *,
    ensure_deployed: bool = True,
) -> None:
    if ensure_deployed:
        _ensure_uhd_batch_app()
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
