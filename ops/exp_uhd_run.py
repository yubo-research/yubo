from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import click

from common.im import im
from llm.console_observer import SplitConsoleObserver
from llm.line_tee import LineRoutingTee, MultiStreamTee


def run_parsed_uhd_local(
    parsed,
    *,
    cfg: dict[str, Any] | None = None,
    results_dir: str = "results/uhd",
    workers: int = 1,
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
        _run_and_save_single_rep(parsed, cfg=cfg, results_dir=results_dir)
        return
    _run_parsed_uhd_direct(parsed)


def _run_and_save_single_rep(parsed, *, cfg: dict[str, Any] | None, results_dir: str) -> None:
    batch_core = im("ops.uhd_batch_core")

    batch_cfg = dict(cfg or {})
    batch_cfg["num_rounds"] = parsed.num_rounds
    total_timesteps = getattr(parsed, "total_timesteps", None)
    if total_timesteps is not None:
        batch_cfg["total_timesteps"] = total_timesteps

    exp_dir = batch_core._experiment_dir(results_dir, batch_cfg)
    batch_core._write_config(exp_dir, batch_cfg)
    base_seed = int(batch_cfg.get("problem_seed", 18))
    jobs = list(batch_core._gen_missing_reps(exp_dir, 1, base_seed))
    if not jobs:
        click.echo(f"All 1 reps done in {exp_dir}")
        return

    _, problem_seed, noise_seed_0, tp = jobs[0]

    buf = io.StringIO()
    if sys.stdout.isatty():
        observer = SplitConsoleObserver(stream=sys.stdout, log_dir=exp_dir)
        tee = _run_capture_stream(sys.stdout, buf, observer.route_line)
        with observer, redirect_stdout(tee), redirect_stderr(sys.stderr):
            _run_parsed_uhd_direct(parsed)
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


def _run_capture_stream(proxy_stream, raw_stream, route_line):
    class _Capture:
        def __init__(self):
            self._router = LineRoutingTee(proxy_stream, route_line, echo=True)

        def write(self, data):
            raw_stream.write(data)
            raw_stream.flush()
            self._router.write(data)

        def flush(self):
            self._router.flush()
            raw_stream.flush()

        def fileno(self):
            return proxy_stream.fileno()

        def isatty(self):
            return proxy_stream.isatty()

        def writable(self):
            method = getattr(proxy_stream, "writable", None)
            return bool(method()) if method is not None else True

        @property
        def encoding(self):
            return getattr(proxy_stream, "encoding", None)

        @property
        def errors(self):
            return getattr(proxy_stream, "errors", None)

    return _Capture()


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
