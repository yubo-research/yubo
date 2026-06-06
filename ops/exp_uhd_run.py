from __future__ import annotations

import click

from common.im import im


def run_parsed_uhd_local(parsed) -> None:
    if parsed.optimizer == "bszo":
        _run_bszo(parsed)
    else:
        _run_unified(parsed)


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


def _run_unified(parsed) -> None:
    make_loop = im("ops.uhd_setup_make_loop").make_loop

    policy_tag = getattr(parsed, "policy_tag", None)
    loop = make_loop(
        parsed.env_tag,
        parsed.num_rounds,
        policy_tag=policy_tag,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        optimizer=parsed.optimizer,
        lr=parsed.lr,
        sigma=parsed.sigma,
        num_dim_target=parsed.num_dim_target,
        num_module_target=parsed.num_module_target,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        early_reject=parsed.early_reject,
        enn=parsed.enn,
        be=parsed.be,
    )
    loop.run()


def uhd_config_toml_to_modal_log(
    config_toml: str,
    gpu: str = "A100",
    *,
    exp_uhd_parse=None,
    tomllib=None,
    modal_run=None,
) -> str:
    p = exp_uhd_parse if exp_uhd_parse is not None else im("ops.exp_uhd_parse")
    if modal_run is None:
        modal_run = im("ops.modal_uhd_runner_impl").run
    cfg = p._load_toml_config(config_toml)
    p._validate_required(cfg)
    parsed = p._parse_cfg(cfg)
    return modal_run(
        parsed.env_tag,
        parsed.num_rounds,
        parsed.lr,
        parsed.num_dim_target,
        parsed.num_module_target,
        policy_tag=parsed.policy_tag,
        gpu=gpu,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        early_reject=parsed.early_reject,
        enn=parsed.enn,
        optimizer=parsed.optimizer,
        sigma=parsed.sigma,
        be=parsed.be,
    )
