#!/usr/bin/env python3

import sys
import tomllib
from pathlib import Path
from typing import Any

import click


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_path()

from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig


_REQUIRED_TOML_KEYS = ("env_tag",)
_OPTIONAL_TOML_KEYS = (
    "num_rounds",
    "total_timesteps",
    "policy_tag",
    "problem_seed",
    "noise_seed_0",
    "lr",
    "sigma",
    "perturb",
    "log_interval",
    "accuracy_interval",
    "target_accuracy",
    # Replication count metadata. Used by ops.uhd_batch; ignored by single-run local/modal commands.
    "num_reps",
    # Optimizer type: "mezo" (default), "simple", "simple_be", "mezo_be", "bszo";
    # UHD vector objectives also support "bszo_be".
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
    "enn_err_ema_beta",
    "enn_max_abs_err_ema",
    "enn_min_calib_points",
    "batch_size",
    # BSZO settings (used when optimizer = "bszo")
    "bszo_k",
    "bszo_epsilon",
    "bszo_sigma_p_sq",
    "bszo_sigma_e_sq",
    "bszo_alpha",
    # JAX/EggRoll evaluator settings (used when env_tag is an EggRoll JAX env).
    "steps_per_episode",
    "eval_episodes",
    "deterministic_policy",
    "seed_offset",
    # Real HyperscaleES pretraining objective settings.
    "pretrain_search_dim",
    "pretrain_delta_scale",
    "pretrain_generation_length",
    "pretrain_rwkv_type",
    "pretrain_lora_only",
    "pretrain_basis_max_leaves",
    # Text generation objective settings for env_tag = "llm:*".
    "max_tokens",
    "temperature",
    "samples_per_prompt",
    "prompt_batch_size",
    "pass_at_k",
    "num_gpus",
    "num_engines",
    "tensor_parallel_size",
    "sub_dataset_size",
    "hf_home",
    "text_search_dim",
    "text_delta_scale",
    "text_basis_max_tensors",
    # Optional upstream EggRoll perturbation materialization for compatible UHD vector objectives.
    "eggroll_noiser",
    "eggroll_rank",
    "eggroll_group_size",
    "eggroll_freeze_nonlora",
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
    "enn_err_ema_beta": 0.95,
    "enn_max_abs_err_ema": 0.25,
    "enn_min_calib_points": 10,
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


def _parse_override_value(raw: str) -> Any:
    from common.config_toml import parse_value

    return parse_value(raw)


def _parse_overrides(override_strings: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in override_strings:
        if "=" not in s:
            raise ValueError(f"Override must be key=value, got: {s}")
        key_raw, value_raw = s.split("=", 1)
        norm = _normalize_key(key_raw.strip())
        if norm not in _ALL_TOML_KEYS:
            raise ValueError(f"Unknown override key '{key_raw}'. Valid keys: {sorted(_ALL_TOML_KEYS)}")
        out[norm] = _parse_override_value(value_raw.strip())
    return out


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_TOML_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_TOML_KEYS)}")
    if "num_rounds" not in cfg and "total_timesteps" not in cfg:
        raise ValueError("Missing required budget field: one of ['num_rounds', 'total_timesteps']")


@click.group()
def _cli():
    pass


cli = _cli


def _parse_perturb(perturb: str) -> tuple[float | None, float | None]:
    """Parse --perturb flag into (num_dim_target, num_module_target)."""
    _backend, ndt, nmt = _parse_perturb_spec(perturb)
    return ndt, nmt


def _parse_perturb_spec(perturb: str) -> tuple[str, float | None, float | None]:
    """Parse --perturb flag into (backend, num_dim_target, num_module_target)."""
    if perturb == "dense":
        return "flat", None, None
    if perturb == "eggroll":
        return "eggroll", None, None
    if perturb.startswith("dim:"):
        return "flat", float(perturb[4:]), None
    if perturb.startswith("mod:"):
        return "flat", None, float(perturb[4:])
    msg = f"Invalid --perturb value: {perturb!r}. Use 'dense', 'dim:<n>', 'mod:<n>', or 'eggroll'."
    raise click.BadParameter(msg)


def _uhd_evals_per_round(optimizer: str, *, bszo_k: int) -> int:
    if optimizer in {"bszo", "bszo_be"}:
        return int(bszo_k) + 1
    return 1


def _derive_num_rounds_from_total_timesteps(
    *,
    env_tag: str,
    total_timesteps: int,
    optimizer: str,
    steps_per_episode: int,
    eval_episodes: int,
    bszo_k: int,
) -> int:
    from problems.uhd_obj import supports_uhd_vector_objective

    if not supports_uhd_vector_objective(env_tag):
        raise ValueError("total_timesteps is currently supported only for UHD vector objective env_tag values; use num_rounds for this config.")
    eval_steps = int(steps_per_episode) * int(eval_episodes)
    round_steps = eval_steps * _uhd_evals_per_round(
        str(optimizer),
        bszo_k=int(bszo_k),
    )
    if round_steps < 1:
        raise ValueError("Cannot derive num_rounds from total_timesteps because per-round step cost is < 1.")
    num_rounds = int(total_timesteps) // int(round_steps)
    if num_rounds < 1:
        raise ValueError(f"total_timesteps={total_timesteps} is smaller than one UHD round cost ({round_steps} env steps).")
    return num_rounds


def _parse_budget_fields(
    cfg: dict[str, Any],
    *,
    env_tag: str,
    optimizer: str,
    steps_per_episode: int,
    eval_episodes: int,
    bszo_k: int,
) -> tuple[int, int | None]:
    total_timesteps = cfg.get("total_timesteps")
    if total_timesteps is not None:
        total_timesteps = int(total_timesteps)
        if total_timesteps < 1:
            raise ValueError(f"total_timesteps must be >= 1 (got: {total_timesteps})")

    num_rounds = cfg.get("num_rounds")
    if num_rounds is not None:
        num_rounds = int(num_rounds)
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 (got: {num_rounds})")
        return num_rounds, total_timesteps

    if total_timesteps is None:
        raise ValueError("Either num_rounds or total_timesteps must be provided.")
    return (
        _derive_num_rounds_from_total_timesteps(
            env_tag=env_tag,
            total_timesteps=total_timesteps,
            optimizer=optimizer,
            steps_per_episode=steps_per_episode,
            eval_episodes=eval_episodes,
            bszo_k=bszo_k,
        ),
        total_timesteps,
    )


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
    err_ema_beta = float(cfg.get("enn_err_ema_beta", _ENN_DEFAULTS["enn_err_ema_beta"]))
    max_abs_err_ema = float(cfg.get("enn_max_abs_err_ema", _ENN_DEFAULTS["enn_max_abs_err_ema"]))
    min_calib_points = int(cfg.get("enn_min_calib_points", _ENN_DEFAULTS["enn_min_calib_points"]))
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
        err_ema_beta=err_ema_beta,
        max_abs_err_ema=max_abs_err_ema,
        min_calib_points=min_calib_points,
    )


def _optional_value(cfg: dict[str, Any], key: str, cast):
    value = cfg.get(key, None)
    return None if value is None else cast(value)


def _parse_core_config_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    perturb_backend, ndt, nmt = _parse_perturb_spec(str(cfg.get("perturb", "dim:0.5")))
    target_accuracy = _optional_value(cfg, "target_accuracy", float)
    return {
        "env_tag": str(cfg["env_tag"]),
        "policy_tag": _optional_value(cfg, "policy_tag", str),
        "num_reps": int(cfg.get("num_reps", 1)),
        "problem_seed": _optional_value(cfg, "problem_seed", int),
        "noise_seed_0": _optional_value(cfg, "noise_seed_0", int),
        "lr": float(cfg.get("lr", 0.001)),
        "sigma": float(cfg.get("sigma", 0.001)),
        "perturb_backend": perturb_backend,
        "num_dim_target": ndt,
        "num_module_target": nmt,
        "log_interval": int(cfg.get("log_interval", 1)),
        "accuracy_interval": int(cfg.get("accuracy_interval", 1000)),
        "target_accuracy": target_accuracy,
        "optimizer": str(cfg.get("optimizer", "mezo")),
        "batch_size": int(cfg.get("batch_size", 4096)),
    }


def _parse_bszo_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "bszo_k": int(cfg.get("bszo_k", 2)),
        "bszo_epsilon": float(cfg.get("bszo_epsilon", 1e-4)),
        "bszo_sigma_p_sq": float(cfg.get("bszo_sigma_p_sq", 1.0)),
        "bszo_sigma_e_sq": float(cfg.get("bszo_sigma_e_sq", 1.0)),
        "bszo_alpha": float(cfg.get("bszo_alpha", 0.1)),
    }


def _parse_eggroll_eval_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps_per_episode": int(cfg.get("steps_per_episode", 200)),
        "eval_episodes": int(cfg.get("eval_episodes", 1)),
        "deterministic_policy": bool(cfg.get("deterministic_policy", False)),
        "seed_offset": int(cfg.get("seed_offset", 0)),
    }


def _parse_pretrain_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    generation_length = _optional_value(cfg, "pretrain_generation_length", int)
    rwkv_type = _optional_value(cfg, "pretrain_rwkv_type", str)
    basis_raw = cfg.get("pretrain_basis_max_leaves", 32)
    fields = {
        "pretrain_search_dim": int(cfg.get("pretrain_search_dim", 4096)),
        "pretrain_delta_scale": float(cfg.get("pretrain_delta_scale", 1.0)),
        "pretrain_generation_length": generation_length,
        "pretrain_rwkv_type": rwkv_type,
        "pretrain_lora_only": bool(cfg.get("pretrain_lora_only", True)),
        "pretrain_basis_max_leaves": None if basis_raw is None or int(basis_raw) <= 0 else int(basis_raw),
    }
    _validate_pretrain_fields(fields)
    return fields


def _parse_text_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    basis_raw = cfg.get("text_basis_max_tensors", 32)
    fields = {
        "max_tokens": int(cfg.get("max_tokens", 1024)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "samples_per_prompt": int(cfg.get("samples_per_prompt", 1)),
        "prompt_batch_size": int(cfg.get("prompt_batch_size", 2)),
        "pass_at_k": bool(cfg.get("pass_at_k", False)),
        "num_gpus": _optional_value(cfg, "num_gpus", int),
        "num_engines": _optional_value(cfg, "num_engines", int),
        "tensor_parallel_size": _optional_value(cfg, "tensor_parallel_size", int),
        "sub_dataset_size": _optional_value(cfg, "sub_dataset_size", int),
        "hf_home": _optional_value(cfg, "hf_home", str),
        "text_search_dim": int(cfg.get("text_search_dim", 256)),
        "text_delta_scale": float(cfg.get("text_delta_scale", 1.0)),
        "text_basis_max_tensors": None if basis_raw is None or int(basis_raw) <= 0 else int(basis_raw),
    }
    _validate_text_fields(fields)
    return fields


def _parse_eggroll_perturb_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "eggroll_noiser": str(cfg.get("eggroll_noiser", "eggroll")),
        "eggroll_rank": int(cfg.get("eggroll_rank", 1)),
        "eggroll_group_size": int(cfg.get("eggroll_group_size", 0)),
        "eggroll_freeze_nonlora": bool(cfg.get("eggroll_freeze_nonlora", False)),
    }


def _validate_pretrain_fields(fields: dict[str, Any]) -> None:
    if fields["pretrain_search_dim"] < 1:
        raise ValueError(f"pretrain_search_dim must be >= 1 (got: {fields['pretrain_search_dim']})")
    if fields["pretrain_delta_scale"] <= 0.0:
        raise ValueError(f"pretrain_delta_scale must be > 0 (got: {fields['pretrain_delta_scale']})")
    if fields["pretrain_generation_length"] is not None and fields["pretrain_generation_length"] < 1:
        raise ValueError(f"pretrain_generation_length must be >= 1 when set (got: {fields['pretrain_generation_length']})")


def _validate_text_fields(fields: dict[str, Any]) -> None:
    if fields["max_tokens"] < 1:
        raise ValueError(f"max_tokens must be >= 1 (got: {fields['max_tokens']})")
    if fields["samples_per_prompt"] < 1:
        raise ValueError(f"samples_per_prompt must be >= 1 (got: {fields['samples_per_prompt']})")
    if fields["prompt_batch_size"] < 1:
        raise ValueError(f"prompt_batch_size must be >= 1 (got: {fields['prompt_batch_size']})")
    if fields["pass_at_k"] and fields["samples_per_prompt"] <= 1:
        raise ValueError("pass_at_k=true requires samples_per_prompt > 1.")
    if fields["num_gpus"] is not None and fields["num_gpus"] < 1:
        raise ValueError(f"num_gpus must be >= 1 when set (got: {fields['num_gpus']})")
    if fields["num_engines"] is not None and fields["num_engines"] < 1:
        raise ValueError(f"num_engines must be >= 1 when set (got: {fields['num_engines']})")
    if fields["tensor_parallel_size"] is not None and fields["tensor_parallel_size"] < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1 when set (got: {fields['tensor_parallel_size']})")
    if fields["sub_dataset_size"] is not None and fields["sub_dataset_size"] < 1:
        raise ValueError(f"sub_dataset_size must be >= 1 when set (got: {fields['sub_dataset_size']})")
    if fields["text_search_dim"] < 1:
        raise ValueError(f"text_search_dim must be >= 1 (got: {fields['text_search_dim']})")
    if fields["text_delta_scale"] <= 0.0:
        raise ValueError(f"text_delta_scale must be > 0 (got: {fields['text_delta_scale']})")


def _parse_cfg(cfg: dict[str, Any]) -> UHDConfig:
    fields = _parse_core_config_fields(cfg)
    fields.update(_parse_bszo_fields(cfg))
    fields.update(_parse_eggroll_eval_fields(cfg))
    fields.update(_parse_pretrain_fields(cfg))
    fields.update(_parse_text_fields(cfg))
    fields.update(_parse_eggroll_perturb_fields(cfg))
    num_rounds, total_timesteps = _parse_budget_fields(
        cfg,
        env_tag=fields["env_tag"],
        optimizer=fields["optimizer"],
        steps_per_episode=fields["steps_per_episode"],
        eval_episodes=fields["eval_episodes"],
        bszo_k=fields["bszo_k"],
    )
    fields.update(
        num_rounds=num_rounds,
        total_timesteps=total_timesteps,
        early_reject=_parse_early_reject_fields(cfg),
        be=_parse_be_fields(cfg),
        enn=_parse_enn_fields(cfg),
    )
    return UHDConfig(**fields)


@_cli.command(name="local", help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "-o",
    "--opt",
    "overrides",
    multiple=True,
    help="Override config key: --opt key=value (e.g. --opt env_tag=quadruped-run-64x64 --opt optimizer=simple)",
)
@click.option("--workers", type=int, default=1, help="Parallel workers when [uhd].num_reps > 1.")
@click.option("--results-dir", type=str, default="results/uhd", help="Results directory when [uhd].num_reps > 1.")
def _local(config_toml: str, overrides: tuple[str, ...] = (), workers: int = 1, results_dir: str = "results/uhd") -> None:
    try:
        cfg = _load_toml_config(config_toml)
        if overrides:
            override_dict = _parse_overrides(overrides)
            cfg = {**cfg, **override_dict}
        _validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = _parse_cfg(cfg)
    if parsed.num_reps > 1:
        from ops.uhd_batch import _batch_local

        batch_cfg = dict(cfg)
        batch_cfg["num_rounds"] = parsed.num_rounds
        if parsed.total_timesteps is not None:
            batch_cfg["total_timesteps"] = parsed.total_timesteps
        _batch_local(batch_cfg, parsed.num_reps, results_dir, workers)
        return
    _ns: dict = {}
    exec("from problems.environment_spec import needs_atari_dm_bindings", _ns)  # noqa: S102
    if _ns["needs_atari_dm_bindings"](parsed.env_tag):
        exec("from problems.env_conf_backends import register_with_env_conf", _ns)  # noqa: S102
        _ns["register_with_env_conf"]()
    _run_parsed(parsed)


def _run_parsed(parsed: UHDConfig) -> None:
    from problems.uhd_obj import supports_uhd_vector_objective

    if supports_uhd_vector_objective(parsed.env_tag):
        from ops.vec_uhd import run_uhd_vector_loop

        run_uhd_vector_loop(parsed)
    elif parsed.optimizer == "bszo":
        _run_bszo(parsed)
    elif parsed.optimizer == "bszo_be":
        raise ValueError("optimizer='bszo_be' is currently only supported for UHD vector objective env_tag values.")
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
        policy_tag=parsed.policy_tag,
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
        parsed.sigma,
        parsed.optimizer,
        policy_tag=parsed.policy_tag,
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
        policy_tag=parsed.policy_tag,
        problem_seed=parsed.problem_seed,
        noise_seed_0=parsed.noise_seed_0,
        batch_size=parsed.batch_size,
        lr=parsed.lr,
        sigma=parsed.sigma,
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


def modal_cmd(
    config_toml: str,
    overrides: tuple[str, ...] = (),
    log_file: str | None = None,
    gpu: str = "A100",
) -> None:
    from ops.modal_uhd import run as modal_run

    try:
        cfg = _load_toml_config(config_toml)
        if overrides:
            override_dict = _parse_overrides(overrides)
            cfg = {**cfg, **override_dict}
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
        sigma=parsed.sigma,
        gpu=gpu,
        policy_tag=parsed.policy_tag,
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


@_cli.command(
    name="modal",
    help="Run on Modal. Streams to stdout; optionally saves to --log-file.",
)
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "-o",
    "--opt",
    "overrides",
    multiple=True,
    help="Override config key: --opt key=value (e.g. --opt env_tag=quadruped-run-64x64)",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also save log to this local file.",
)
@click.option("--gpu", type=str, default="A100", help="Modal GPU type (e.g. T4, A10, A100, H100).")
def _modal_cmd_cli(config_toml: str, overrides: tuple[str, ...], log_file: str | None, gpu: str) -> None:
    modal_cmd(config_toml, overrides, log_file, gpu)


if __name__ == "__main__":
    cli()
