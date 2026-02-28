#!/usr/bin/env python

from typing import Any

import click
import tomllib

from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig

_REQUIRED_TOML_KEYS = ("env_tag", "num_rounds")
_OPTIONAL_TOML_KEYS = (
    "problem_seed",
    "noise_seed_0",
    "lr",
    "perturb",
    "log_interval",
    "accuracy_interval",
    "target_accuracy",
    # Optimizer type: "mezo" (default), "simple", or "simple_be"
    "optimizer",
    # BehavioralEmbedder settings (used when optimizer = "simple_be")
    "be_num_probes",
    "be_num_candidates",
    "be_warmup",
    "be_fit_interval",
    "be_enn_k",
    "be_sigma_range",
    # ENN minus imputation (prototype)
    "enn_minus_impute",
    "enn_d",
    "enn_s",
    "enn_jl_seed",
    "enn_k",
    "enn_fit_interval",
    "enn_warmup_real_obs",
    "enn_refresh_interval",
    "enn_se_threshold",
    "enn_target",
    "enn_num_candidates",
    "enn_select_interval",
    "enn_embedder",
    "enn_gather_t",
    "batch_size",
    # BSZO settings (used when optimizer = "bszo")
    "bszo_k",
    "bszo_epsilon",
    "bszo_sigma_p_sq",
    "bszo_sigma_e_sq",
    "bszo_alpha",
)
_ALL_TOML_KEYS = set(_REQUIRED_TOML_KEYS + _OPTIONAL_TOML_KEYS)

_ENN_DEFAULTS: dict[str, object] = {
    "enn_minus_impute": False,
    "enn_d": 100,
    "enn_s": 4,
    "enn_jl_seed": 123,
    "enn_k": 25,
    "enn_fit_interval": 50,
    "enn_warmup_real_obs": 200,
    "enn_refresh_interval": 50,
    "enn_se_threshold": 0.25,
    "enn_target": "mu_minus",
    "enn_num_candidates": 1,
    "enn_select_interval": 1,
    "enn_embedder": "direction",
    "enn_gather_t": 64,
}

_ER_DEFAULTS: dict[str, object] = {
    "er_tau": None,
    "er_mode": None,
    "er_ema_beta": None,
    "er_warmup_pos": None,
    "er_quantile": None,
    "er_window": None,
}

_BE_DEFAULTS: dict[str, object] = {
    "be_num_probes": 10,
    "be_num_candidates": 10,
    "be_warmup": 20,
    "be_fit_interval": 10,
    "be_enn_k": 25,
    "be_sigma_range": None,
}


def _normalize_key(key: str) -> str:
    # Match experiments/experiment.py convention: allow hyphenated keys in TOML.
    return key.replace("-", "_")


def _coerce_mapping_keys(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("TOML config must be a mapping at root or under [uhd].")

    out: dict[str, Any] = {}
    for key, value in raw.items():
        norm = _normalize_key(str(key))
        if norm not in _ALL_TOML_KEYS:
            raise ValueError(f"Unknown key '{key}' in {source}. Valid keys: {sorted(_ALL_TOML_KEYS)}")
        out[norm] = value
    return out


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("uhd", data)
    return _coerce_mapping_keys(section, source=f"TOML '{path}'")


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_TOML_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_TOML_KEYS)}")


@click.group()
def _cli():
    pass


cli = _cli


def _parse_perturb(perturb: str) -> tuple[float | None, float | None]:
    """Parse --perturb flag into (num_dim_target, num_module_target)."""
    if perturb == "dense":
        return None, None
    if perturb.startswith("dim:"):
        return float(perturb[4:]), None
    if perturb.startswith("mod:"):
        return None, float(perturb[4:])
    msg = f"Invalid --perturb value: {perturb!r}. Use 'dense', 'dim:<n>', or 'mod:<n>'."
    raise click.BadParameter(msg)


def _parse_early_reject_fields(cfg: dict[str, Any]) -> EarlyRejectConfig:
    tau = cfg.get("er_tau", _ER_DEFAULTS["er_tau"])
    if tau is not None:
        tau = float(tau)
    mode = cfg.get("er_mode", _ER_DEFAULTS["er_mode"])
    if mode is not None:
        mode = str(mode)
    ema_beta = cfg.get("er_ema_beta", _ER_DEFAULTS["er_ema_beta"])
    if ema_beta is not None:
        ema_beta = float(ema_beta)
    warmup_pos = cfg.get("er_warmup_pos", _ER_DEFAULTS["er_warmup_pos"])
    if warmup_pos is not None:
        warmup_pos = int(warmup_pos)
    quantile = cfg.get("er_quantile", _ER_DEFAULTS["er_quantile"])
    if quantile is not None:
        quantile = float(quantile)
    window = cfg.get("er_window", _ER_DEFAULTS["er_window"])
    if window is not None:
        window = int(window)
    return EarlyRejectConfig(
        tau=tau,
        mode=mode,
        ema_beta=ema_beta,
        warmup_pos=warmup_pos,
        quantile=quantile,
        window=window,
    )


def _parse_be_fields(cfg: dict[str, Any]) -> BEConfig:
    num_probes = int(cfg.get("be_num_probes", _BE_DEFAULTS["be_num_probes"]))
    num_candidates = int(cfg.get("be_num_candidates", _BE_DEFAULTS["be_num_candidates"]))
    warmup = int(cfg.get("be_warmup", _BE_DEFAULTS["be_warmup"]))
    fit_interval = int(cfg.get("be_fit_interval", _BE_DEFAULTS["be_fit_interval"]))
    enn_k = int(cfg.get("be_enn_k", _BE_DEFAULTS["be_enn_k"]))
    sigma_range_raw = cfg.get("be_sigma_range", _BE_DEFAULTS["be_sigma_range"])
    sigma_range = tuple(float(v) for v in sigma_range_raw) if sigma_range_raw is not None else None
    return BEConfig(
        num_probes=num_probes,
        num_candidates=num_candidates,
        warmup=warmup,
        fit_interval=fit_interval,
        enn_k=enn_k,
        sigma_range=sigma_range,
    )


def _parse_enn_fields(cfg: dict[str, Any]) -> ENNConfig:
    minus_impute = bool(cfg.get("enn_minus_impute", _ENN_DEFAULTS["enn_minus_impute"]))
    d = int(cfg.get("enn_d", _ENN_DEFAULTS["enn_d"]))
    s = int(cfg.get("enn_s", _ENN_DEFAULTS["enn_s"]))
    jl_seed = int(cfg.get("enn_jl_seed", _ENN_DEFAULTS["enn_jl_seed"]))
    k = int(cfg.get("enn_k", _ENN_DEFAULTS["enn_k"]))
    fit_interval = int(cfg.get("enn_fit_interval", _ENN_DEFAULTS["enn_fit_interval"]))
    warmup_real_obs = int(cfg.get("enn_warmup_real_obs", _ENN_DEFAULTS["enn_warmup_real_obs"]))
    refresh_interval = int(cfg.get("enn_refresh_interval", _ENN_DEFAULTS["enn_refresh_interval"]))
    se_threshold = float(cfg.get("enn_se_threshold", _ENN_DEFAULTS["enn_se_threshold"]))
    target = str(cfg.get("enn_target", _ENN_DEFAULTS["enn_target"]))
    num_candidates = int(cfg.get("enn_num_candidates", _ENN_DEFAULTS["enn_num_candidates"]))
    select_interval = int(cfg.get("enn_select_interval", _ENN_DEFAULTS["enn_select_interval"]))
    embedder = str(cfg.get("enn_embedder", _ENN_DEFAULTS["enn_embedder"]))
    gather_t = int(cfg.get("enn_gather_t", _ENN_DEFAULTS["enn_gather_t"]))
    return ENNConfig(
        minus_impute=minus_impute,
        d=d,
        s=s,
        jl_seed=jl_seed,
        k=k,
        fit_interval=fit_interval,
        warmup_real_obs=warmup_real_obs,
        refresh_interval=refresh_interval,
        se_threshold=se_threshold,
        target=target,
        num_candidates=num_candidates,
        select_interval=select_interval,
        embedder=embedder,
        gather_t=gather_t,
    )


def _parse_cfg(cfg: dict[str, Any]) -> UHDConfig:
    env_tag = str(cfg["env_tag"])
    num_rounds = int(cfg["num_rounds"])
    problem_seed = cfg.get("problem_seed", None)
    if problem_seed is not None:
        problem_seed = int(problem_seed)
    noise_seed_0 = cfg.get("noise_seed_0", None)
    if noise_seed_0 is not None:
        noise_seed_0 = int(noise_seed_0)
    lr = float(cfg.get("lr", 0.001))
    perturb = str(cfg.get("perturb", "dim:0.5"))
    log_interval = int(cfg.get("log_interval", 1))
    accuracy_interval = int(cfg.get("accuracy_interval", 1000))
    target_accuracy = cfg.get("target_accuracy", None)
    if target_accuracy is not None:
        target_accuracy = float(target_accuracy)
    optimizer = str(cfg.get("optimizer", "mezo"))
    batch_size = int(cfg.get("batch_size", 4096))
    bszo_k = int(cfg.get("bszo_k", 2))
    bszo_epsilon = float(cfg.get("bszo_epsilon", 1e-4))
    bszo_sigma_p_sq = float(cfg.get("bszo_sigma_p_sq", 1.0))
    bszo_sigma_e_sq = float(cfg.get("bszo_sigma_e_sq", 1.0))
    bszo_alpha = float(cfg.get("bszo_alpha", 0.1))
    early_reject = _parse_early_reject_fields(cfg)
    be = _parse_be_fields(cfg)
    enn = _parse_enn_fields(cfg)
    ndt, nmt = _parse_perturb(perturb)
    return UHDConfig(
        env_tag=env_tag,
        num_rounds=num_rounds,
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        lr=lr,
        num_dim_target=ndt,
        num_module_target=nmt,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
        optimizer=optimizer,
        batch_size=batch_size,
        early_reject=early_reject,
        be=be,
        enn=enn,
        bszo_k=bszo_k,
        bszo_epsilon=bszo_epsilon,
        bszo_sigma_p_sq=bszo_sigma_p_sq,
        bszo_sigma_e_sq=bszo_sigma_e_sq,
        bszo_alpha=bszo_alpha,
    )


@_cli.command(name="local", help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
def _local(config_toml: str) -> None:
    try:
        cfg = _load_toml_config(config_toml)
        _validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = _parse_cfg(cfg)
    _run_parsed(parsed)


def _run_parsed(parsed: UHDConfig) -> None:
    if parsed.optimizer == "bszo":
        _run_bszo(parsed)
    elif parsed.optimizer in {"simple", "simple_be", "mezo_be"}:
        _run_simple(parsed)
    else:
        _run_mezo(parsed)


def _run_bszo(parsed: UHDConfig) -> None:
    from ops.uhd_setup import run_bszo_loop

    run_bszo_loop(
        parsed.env_tag,
        parsed.num_rounds,
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


def _run_simple(parsed: UHDConfig) -> None:
    from ops.uhd_setup import run_simple_loop

    run_simple_loop(
        parsed.env_tag,
        parsed.num_rounds,
        0.001,
        parsed.optimizer,
        num_dim_target=parsed.num_dim_target,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        be=parsed.be,
    )


def _run_mezo(parsed: UHDConfig) -> None:
    from ops.uhd_setup import make_loop

    loop = make_loop(
        parsed.env_tag,
        parsed.num_rounds,
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


local = _local


@_cli.command(
    name="modal",
    help="Run on Modal. Streams to stdout; optionally saves to --log-file.",
)
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also save log to this local file.",
)
@click.option("--gpu", type=str, default="A100", help="Modal GPU type (e.g. T4, A10, A100, H100).")
def modal_cmd(config_toml: str, log_file: str | None, gpu: str) -> None:
    from ops.modal_uhd import run as modal_run

    try:
        cfg = _load_toml_config(config_toml)
        _validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = _parse_cfg(cfg)
    log_text = modal_run(
        parsed.env_tag,
        parsed.num_rounds,
        parsed.lr,
        parsed.num_dim_target,
        parsed.num_module_target,
        gpu=gpu,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        log_interval=parsed.log_interval,
        accuracy_interval=parsed.accuracy_interval,
        target_accuracy=parsed.target_accuracy,
        early_reject=parsed.early_reject,
        enn=parsed.enn,
    )

    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(log_text)
        click.echo(f"Log saved to {log_file}")


if __name__ == "__main__":
    cli()
