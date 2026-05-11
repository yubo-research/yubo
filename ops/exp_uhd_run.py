from __future__ import annotations

from typing import Any

import click

from common.im import im


def run_parsed_uhd_local(parsed, *, cfg: dict[str, Any] | None = None, results_dir: str = "results/uhd", workers: int = 1) -> None:
    if getattr(parsed, "num_reps", 1) > 1:
        batch_cfg = dict(cfg or {})
        batch_cfg["num_rounds"] = parsed.num_rounds
        total_timesteps = getattr(parsed, "total_timesteps", None)
        if total_timesteps is not None:
            batch_cfg["total_timesteps"] = total_timesteps
        im("ops.uhd_batch")._batch_local(batch_cfg, parsed.num_reps, results_dir, workers)
        return
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
