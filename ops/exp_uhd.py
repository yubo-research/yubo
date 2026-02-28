#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

import click
import tomllib

_REQUIRED_TOML_KEYS = ("env_tag", "num_rounds")
_OPTIONAL_TOML_KEYS = (
    "problem_seed",
    "noise_seed_0",
    "lr",
    "perturb",
    "log_interval",
    "accuracy_interval",
    "target_accuracy",
    "early_reject_tau",
    "early_reject_mode",
    "early_reject_ema_beta",
    "early_reject_warmup_pos",
    "early_reject_quantile",
    "early_reject_window",
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


@dataclass(frozen=True)
class UHDConfig:
    env_tag: str
    num_rounds: int
    problem_seed: int | None
    noise_seed_0: int | None
    lr: float
    num_dim_target: float | None
    num_module_target: float | None
    log_interval: int
    accuracy_interval: int
    target_accuracy: float | None
    early_reject_tau: float | None
    early_reject_mode: str | None
    early_reject_ema_beta: float | None
    early_reject_warmup_pos: int | None
    early_reject_quantile: float | None
    early_reject_window: int | None
    optimizer: str
    be_num_probes: int
    be_num_candidates: int
    be_warmup: int
    be_fit_interval: int
    be_enn_k: int
    be_sigma_range: tuple[float, float] | None
    batch_size: int
    enn_minus_impute: bool
    enn_d: int
    enn_s: int
    enn_jl_seed: int
    enn_k: int
    enn_fit_interval: int
    enn_warmup_real_obs: int
    enn_refresh_interval: int
    enn_se_threshold: float
    enn_target: str
    enn_num_candidates: int
    enn_select_interval: int
    enn_embedder: str
    enn_gather_t: int
    bszo_k: int
    bszo_epsilon: float
    bszo_sigma_p_sq: float
    bszo_sigma_e_sq: float
    bszo_alpha: float


@dataclass(frozen=True)
class _EarlyRejectFields:
    tau: float | None
    mode: str | None
    ema_beta: float | None
    warmup_pos: int | None
    quantile: float | None
    window: int | None


def _parse_early_reject_fields(cfg: dict[str, Any]) -> _EarlyRejectFields:
    early_reject_tau = cfg.get("early_reject_tau", None)
    if early_reject_tau is not None:
        early_reject_tau = float(early_reject_tau)
    early_reject_mode = cfg.get("early_reject_mode", None)
    if early_reject_mode is not None:
        early_reject_mode = str(early_reject_mode)
    early_reject_ema_beta = cfg.get("early_reject_ema_beta", None)
    if early_reject_ema_beta is not None:
        early_reject_ema_beta = float(early_reject_ema_beta)
    early_reject_warmup_pos = cfg.get("early_reject_warmup_pos", None)
    if early_reject_warmup_pos is not None:
        early_reject_warmup_pos = int(early_reject_warmup_pos)
    early_reject_quantile = cfg.get("early_reject_quantile", None)
    if early_reject_quantile is not None:
        early_reject_quantile = float(early_reject_quantile)
    early_reject_window = cfg.get("early_reject_window", None)
    if early_reject_window is not None:
        early_reject_window = int(early_reject_window)
    return _EarlyRejectFields(
        tau=early_reject_tau,
        mode=early_reject_mode,
        ema_beta=early_reject_ema_beta,
        warmup_pos=early_reject_warmup_pos,
        quantile=early_reject_quantile,
        window=early_reject_window,
    )


@dataclass(frozen=True)
class _ENNFields:
    minus_impute: bool
    d: int
    s: int
    jl_seed: int
    k: int
    fit_interval: int
    warmup_real_obs: int
    refresh_interval: int
    se_threshold: float
    target: str
    num_candidates: int
    select_interval: int
    embedder: str
    gather_t: int


def _parse_enn_fields(cfg: dict[str, Any]) -> _ENNFields:
    enn_minus_impute = bool(cfg.get("enn_minus_impute", _ENN_DEFAULTS["enn_minus_impute"]))
    enn_d = int(cfg.get("enn_d", _ENN_DEFAULTS["enn_d"]))
    enn_s = int(cfg.get("enn_s", _ENN_DEFAULTS["enn_s"]))
    enn_jl_seed = int(cfg.get("enn_jl_seed", _ENN_DEFAULTS["enn_jl_seed"]))
    enn_k = int(cfg.get("enn_k", _ENN_DEFAULTS["enn_k"]))
    enn_fit_interval = int(cfg.get("enn_fit_interval", _ENN_DEFAULTS["enn_fit_interval"]))
    enn_warmup_real_obs = int(cfg.get("enn_warmup_real_obs", _ENN_DEFAULTS["enn_warmup_real_obs"]))
    enn_refresh_interval = int(cfg.get("enn_refresh_interval", _ENN_DEFAULTS["enn_refresh_interval"]))
    enn_se_threshold = float(cfg.get("enn_se_threshold", _ENN_DEFAULTS["enn_se_threshold"]))
    enn_target = str(cfg.get("enn_target", _ENN_DEFAULTS["enn_target"]))
    enn_num_candidates = int(cfg.get("enn_num_candidates", _ENN_DEFAULTS["enn_num_candidates"]))
    enn_select_interval = int(cfg.get("enn_select_interval", _ENN_DEFAULTS["enn_select_interval"]))
    enn_embedder = str(cfg.get("enn_embedder", _ENN_DEFAULTS["enn_embedder"]))
    enn_gather_t = int(cfg.get("enn_gather_t", _ENN_DEFAULTS["enn_gather_t"]))
    return _ENNFields(
        minus_impute=enn_minus_impute,
        d=enn_d,
        s=enn_s,
        jl_seed=enn_jl_seed,
        k=enn_k,
        fit_interval=enn_fit_interval,
        warmup_real_obs=enn_warmup_real_obs,
        refresh_interval=enn_refresh_interval,
        se_threshold=enn_se_threshold,
        target=enn_target,
        num_candidates=enn_num_candidates,
        select_interval=enn_select_interval,
        embedder=enn_embedder,
        gather_t=enn_gather_t,
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
    be_num_probes = int(cfg.get("be_num_probes", 10))
    be_num_candidates = int(cfg.get("be_num_candidates", 10))
    be_warmup = int(cfg.get("be_warmup", 20))
    be_fit_interval = int(cfg.get("be_fit_interval", 10))
    be_enn_k = int(cfg.get("be_enn_k", 25))
    be_sigma_range_raw = cfg.get("be_sigma_range", None)
    be_sigma_range = tuple(float(v) for v in be_sigma_range_raw) if be_sigma_range_raw is not None else None
    batch_size = int(cfg.get("batch_size", 4096))
    bszo_k = int(cfg.get("bszo_k", 2))
    bszo_epsilon = float(cfg.get("bszo_epsilon", 1e-4))
    bszo_sigma_p_sq = float(cfg.get("bszo_sigma_p_sq", 1.0))
    bszo_sigma_e_sq = float(cfg.get("bszo_sigma_e_sq", 1.0))
    bszo_alpha = float(cfg.get("bszo_alpha", 0.1))
    er = _parse_early_reject_fields(cfg)
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
        early_reject_tau=er.tau,
        early_reject_mode=er.mode,
        early_reject_ema_beta=er.ema_beta,
        early_reject_warmup_pos=er.warmup_pos,
        early_reject_quantile=er.quantile,
        early_reject_window=er.window,
        optimizer=optimizer,
        be_num_probes=be_num_probes,
        be_num_candidates=be_num_candidates,
        be_warmup=be_warmup,
        be_fit_interval=be_fit_interval,
        be_enn_k=be_enn_k,
        be_sigma_range=be_sigma_range,
        batch_size=batch_size,
        enn_minus_impute=enn.minus_impute,
        enn_d=enn.d,
        enn_s=enn.s,
        enn_jl_seed=enn.jl_seed,
        enn_k=enn.k,
        enn_fit_interval=enn.fit_interval,
        enn_warmup_real_obs=enn.warmup_real_obs,
        enn_refresh_interval=enn.refresh_interval,
        enn_se_threshold=enn.se_threshold,
        enn_target=enn.target,
        enn_num_candidates=enn.num_candidates,
        enn_select_interval=enn.select_interval,
        enn_embedder=enn.embedder,
        enn_gather_t=enn.gather_t,
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
        be_num_probes=parsed.be_num_probes,
        be_num_candidates=parsed.be_num_candidates,
        be_warmup=parsed.be_warmup,
        be_fit_interval=parsed.be_fit_interval,
        be_enn_k=parsed.be_enn_k,
        be_sigma_range=parsed.be_sigma_range,
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
        early_reject_tau=parsed.early_reject_tau,
        early_reject_mode=parsed.early_reject_mode,
        early_reject_ema_beta=parsed.early_reject_ema_beta,
        early_reject_warmup_pos=parsed.early_reject_warmup_pos,
        early_reject_quantile=parsed.early_reject_quantile,
        early_reject_window=parsed.early_reject_window,
        enn={
            "enn_minus_impute": parsed.enn_minus_impute,
            "enn_d": parsed.enn_d,
            "enn_s": parsed.enn_s,
            "enn_jl_seed": parsed.enn_jl_seed,
            "enn_k": parsed.enn_k,
            "enn_fit_interval": parsed.enn_fit_interval,
            "enn_warmup_real_obs": parsed.enn_warmup_real_obs,
            "enn_refresh_interval": parsed.enn_refresh_interval,
            "enn_se_threshold": parsed.enn_se_threshold,
            "enn_target": parsed.enn_target,
            "enn_num_candidates": parsed.enn_num_candidates,
            "enn_select_interval": parsed.enn_select_interval,
            "enn_embedder": parsed.enn_embedder,
            "enn_gather_t": parsed.enn_gather_t,
        },
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
        early_reject_tau=parsed.early_reject_tau,
        early_reject_mode=parsed.early_reject_mode,
        early_reject_ema_beta=parsed.early_reject_ema_beta,
        early_reject_warmup_pos=parsed.early_reject_warmup_pos,
        early_reject_quantile=parsed.early_reject_quantile,
        early_reject_window=parsed.early_reject_window,
        enn={
            "enn_minus_impute": parsed.enn_minus_impute,
            "enn_d": parsed.enn_d,
            "enn_s": parsed.enn_s,
            "enn_jl_seed": parsed.enn_jl_seed,
            "enn_k": parsed.enn_k,
            "enn_fit_interval": parsed.enn_fit_interval,
            "enn_warmup_real_obs": parsed.enn_warmup_real_obs,
            "enn_refresh_interval": parsed.enn_refresh_interval,
            "enn_se_threshold": parsed.enn_se_threshold,
            "enn_target": parsed.enn_target,
            "enn_num_candidates": parsed.enn_num_candidates,
            "enn_select_interval": parsed.enn_select_interval,
            "enn_embedder": parsed.enn_embedder,
            "enn_gather_t": parsed.enn_gather_t,
        },
    )

    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(log_text)
        click.echo(f"Log saved to {log_file}")


if __name__ == "__main__":
    cli()
