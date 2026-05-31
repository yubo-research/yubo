from __future__ import annotations

import io
import subprocess
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import click

from common.im import im
from llm.console_dashboard import run_console_dashboard
from llm.console_observer import SplitConsoleObserver
from llm.line_tee import MultiStreamTee


def run_parsed_uhd_local(
    parsed,
    *,
    cfg: dict[str, Any] | None = None,
    results_dir: str = "results/uhd",
    workers: int = 1,
    config_toml: str | None = None,
    overrides: tuple[str, ...] = (),
    dashboard: bool = False,
    child_process: bool = False,
) -> None:
    if getattr(parsed, "num_reps", 1) > 1:
        batch_cfg = dict(cfg or {})
        batch_cfg["num_rounds"] = parsed.num_rounds
        total_timesteps = getattr(parsed, "total_timesteps", None)
        if total_timesteps is not None:
            batch_cfg["total_timesteps"] = total_timesteps
        im("ops.uhd_batch")._batch_local(batch_cfg, parsed.num_reps, results_dir, workers)
        return
    if getattr(parsed, "num_reps", 1) == 1:
        _run_and_save_single_rep(
            parsed,
            cfg=cfg,
            results_dir=results_dir,
            config_toml=config_toml,
            overrides=overrides,
            dashboard=dashboard,
            child_process=child_process,
        )
        return
    _run_parsed_uhd_direct(parsed)


def _run_and_save_single_rep(
    parsed,
    *,
    cfg: dict[str, Any] | None,
    results_dir: str,
    config_toml: str | None = None,
    overrides: tuple[str, ...] = (),
    dashboard: bool = False,
    child_process: bool = False,
) -> None:
    batch_core = im("ops.uhd_batch_core")

    batch_cfg = dict(cfg or {})
    batch_cfg["num_rounds"] = parsed.num_rounds
    total_timesteps = getattr(parsed, "total_timesteps", None)
    if total_timesteps is not None:
        batch_cfg["total_timesteps"] = total_timesteps

    use_dashboard = _use_dashboard_console(
        dashboard=dashboard,
        stream=sys.stdout,
        child_process=child_process,
        env_tag=getattr(parsed, "env_tag", None),
    )

    exp_dir = batch_core._experiment_dir(results_dir, batch_cfg)
    batch_core._write_config(exp_dir, batch_cfg)
    base_seed = int(batch_cfg.get("problem_seed", 18))
    jobs = list(batch_core._gen_missing_reps(exp_dir, 1, base_seed))
    if not jobs:
        click.echo(f"All 1 reps done in {exp_dir}")
        return

    _, problem_seed, noise_seed_0, tp = jobs[0]

    buf = io.StringIO()
    if use_dashboard:
        _run_and_save_single_rep_dashboard(
            results_dir=results_dir,
            exp_dir=exp_dir,
            buf=buf,
            config_toml=config_toml,
            overrides=overrides,
        )
    else:
        tee = MultiStreamTee(sys.stdout, buf)
        with redirect_stdout(tee), redirect_stderr(sys.stderr):
            _run_parsed_uhd_direct(parsed)

    records = im("ops.uhd_batch")._parse_eval_lines(buf.getvalue())
    if not records:
        if not buf.getvalue().strip():
            click.echo(f"Single-rep run produced no EVAL records; no trace saved in {exp_dir}")
            return
        raise click.ClickException("Single-rep run completed but produced no EVAL records.")
    batch_core._write_trace(tp, records)
    click.echo(f"Single-rep run saved to {exp_dir}")


def _use_dashboard_console(
    *,
    dashboard: bool,
    stream,
    child_process: bool,
    env_tag: str | None = None,
) -> bool:
    if child_process:
        return False
    if not dashboard:
        return False
    if not str(env_tag or "").startswith("llm:"):
        raise click.ClickException("The UHD dashboard is only available for llm:* envs.")
    return bool(hasattr(stream, "isatty") and stream.isatty())


def _run_and_save_single_rep_dashboard(
    *,
    results_dir: str,
    exp_dir,
    buf: io.StringIO,
    config_toml: str | None,
    overrides: tuple[str, ...],
) -> None:
    observer = SplitConsoleObserver(stream=None, log_dir=exp_dir, enable_tui=False)
    done = threading.Event()
    exc: list[BaseException] = []

    cmd = _child_run_command(config_toml, overrides=overrides, results_dir=results_dir)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _pump(stream, route):
        try:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                clean = line.rstrip("\n")
                buf.write(line)
                route(clean)
        except BaseException as err:  # noqa: BLE001
            exc.append(err)

    with observer:
        stdout_thread = threading.Thread(target=_pump, args=(proc.stdout, observer.route_line), daemon=True)
        stderr_thread = threading.Thread(target=_pump, args=(proc.stderr, observer.append_diagnostics), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        def _wait() -> None:
            rc = proc.wait()
            if rc != 0 and not exc:
                exc.append(click.ClickException(f"UHD child process exited with status {rc}"))
            done.set()

        waiter = threading.Thread(target=_wait, daemon=True)
        waiter.start()
        try:
            run_console_dashboard(observer, title="UHD Console", done_event=done)
        finally:
            done.set()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            waiter.join(timeout=1)
    if exc:
        raise exc[0]


def _child_run_command(
    config_toml: str | None,
    *,
    overrides: tuple[str, ...],
    results_dir: str,
) -> list[str]:
    if not config_toml:
        raise click.ClickException("Internal error: dashboard launch requires config_toml")
    config_toml = str(Path(config_toml).resolve())
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().with_name("exp_uhd.py")),
        "local",
        config_toml,
        "--results-dir",
        results_dir,
        "--child-process",
    ]
    if overrides:
        for override in overrides:
            cmd.extend(["--opt", override])
    return cmd


def _run_parsed_uhd_direct(parsed) -> None:
    _register_runtime_backends(parsed.env_tag)
    supports_uhd_vector_objective = im("problems.uhd_obj").supports_uhd_vector_objective

    if supports_uhd_vector_objective(parsed.env_tag):
        _run_vec_uhd(parsed)
    elif parsed.optimizer == "bszo":
        _run_bszo(parsed)
    elif parsed.optimizer == "bszo_be":
        raise ValueError("optimizer='bszo_be' is currently only supported for UHD vector objective env_tag values.")
    elif parsed.optimizer in {"simple", "simple_be", "mezo_be"}:
        _run_simple(parsed)
    else:
        _run_mezo(parsed)


def _register_runtime_backends(env_tag: str) -> None:
    needs_atari_dm_bindings = im("problems.environment_spec").needs_atari_dm_bindings

    if needs_atari_dm_bindings(env_tag):
        im("problems.env_conf_backends").register_with_env_conf()


def _run_bszo(parsed) -> None:
    policy_tag = getattr(parsed, "policy_tag", None)
    im("ops.uhd_setup_bszo").run_bszo_loop(
        parsed.env_tag,
        parsed.num_rounds,
        policy_tag=policy_tag,
        lr=parsed.lr,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        bszo_k=parsed.bszo_k,
        bszo_epsilon=parsed.bszo_epsilon,
        bszo_sigma_p_sq=parsed.bszo_sigma_p_sq,
        bszo_sigma_e_sq=parsed.bszo_sigma_e_sq,
        bszo_alpha=parsed.bszo_alpha,
    )


def _run_simple(parsed) -> None:
    policy_tag = getattr(parsed, "policy_tag", None)
    im("ops.uhd_setup").run_simple_loop(
        parsed.env_tag,
        parsed.num_rounds,
        getattr(parsed, "sigma", 0.001),
        parsed.optimizer,
        policy_tag=policy_tag,
        num_dim_target=parsed.num_dim_target,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        be=parsed.be,
    )


def _run_vec_uhd(parsed) -> None:
    im("ops.vec_uhd").run_uhd_vector_loop(parsed)


def _run_mezo(parsed) -> None:
    policy_tag = getattr(parsed, "policy_tag", None)
    loop = im("ops.uhd_setup_make_loop").make_loop(
        parsed.env_tag,
        parsed.num_rounds,
        policy_tag=policy_tag,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        lr=parsed.lr,
        sigma=getattr(parsed, "sigma", 0.001),
        num_dim_target=parsed.num_dim_target,
        num_module_target=parsed.num_module_target,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        early_reject=parsed.early_reject,
        enn=parsed.enn,
    )
    loop.run()


def uhd_config_toml_to_modal_log(
    config_toml: str,
    gpu: str,
    *,
    exp_uhd_parse: Any,
    tomllib: Any,
    modal_run: Any,
) -> str:
    try:
        cfg = exp_uhd_parse._load_toml_config(config_toml)
        exp_uhd_parse._validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e
    parsed = exp_uhd_parse._parse_cfg(cfg)
    policy_tag = getattr(parsed, "policy_tag", None)
    return modal_run(
        parsed.env_tag,
        parsed.num_rounds,
        parsed.lr,
        parsed.num_dim_target,
        parsed.num_module_target,
        sigma=getattr(parsed, "sigma", 0.001),
        gpu=gpu,
        policy_tag=policy_tag,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        early_reject=parsed.early_reject,
        enn=parsed.enn,
    )
