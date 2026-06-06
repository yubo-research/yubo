from __future__ import annotations

from typing import Any

import click

from common.im import im


def run_parsed_uhd_local(parsed) -> None:
    if parsed.optimizer == "bszo":
        _run_bszo(parsed)
    elif parsed.optimizer in {"simple", "simple_be", "mezo_be"}:
        _run_simple(parsed)
    else:
        _run_mezo(parsed)


def run_local_from_toml(config_toml: str) -> None:
    tomllib = im("tomllib")
    p = im("ops.exp_uhd_parse")
    try:
        cfg = p._load_toml_config(config_toml)
        p._validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e
    parsed = p._parse_cfg(cfg)
    run_parsed_uhd_local(parsed)


def _run_bszo(parsed) -> None:
    run_bszo_loop = im("ops.uhd_setup_bszo").run_bszo_loop

    policy_tag = getattr(parsed, "policy_tag", None)
    run_bszo_loop(
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
    run_simple_loop = im("ops.uhd_setup_simple_gym").run_simple_loop

    policy_tag = getattr(parsed, "policy_tag", None)
    run_simple_loop(
        parsed.env_tag,
        parsed.num_rounds,
        0.001,
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


def _run_mezo(parsed) -> None:
    make_loop = im("ops.uhd_setup_make_loop").make_loop

    policy_tag = getattr(parsed, "policy_tag", None)
    loop = make_loop(
        parsed.env_tag,
        parsed.num_rounds,
        policy_tag=policy_tag,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        lr=parsed.lr,
        sigma=0.001,
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
