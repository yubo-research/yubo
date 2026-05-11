import tomllib
from typing import Any

import click

from common.mapping_keys import coerce_mapping_keys, normalize_toml_key
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
    "num_reps",
    # Optimizer type: "mezo" (default), "simple", "simple_be", "mezo_be", or "bszo";
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
    # Early-reject
    "er_tau",
    "er_mode",
    "er_ema_beta",
    "er_warmup_pos",
    "er_quantile",
    "er_window",
    # BSZO settings (used when optimizer = "bszo")
    "bszo_k",
    "bszo_epsilon",
    "bszo_sigma_p_sq",
    "bszo_sigma_e_sq",
    "bszo_alpha",
    # JAX/EggRoll evaluator settings.
    "steps_per_episode",
    "num_envs",
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
    # Optional EggRoll-style perturbation materialization for compatible UHD vector objectives.
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
    return normalize_toml_key(key)


def _coerce_mapping_keys(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    return coerce_mapping_keys(
        raw,
        source=source,
        valid_keys=_ALL_TOML_KEYS,
        not_mapping_msg="TOML config must be a mapping at root or under [uhd].",
    )


def _load_toml_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("uhd", data)
    return _coerce_mapping_keys(section, source=f"TOML '{path}'")


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_TOML_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_TOML_KEYS)}")
    if "num_rounds" not in cfg and "total_timesteps" not in cfg:
        raise ValueError("Missing required budget field: one of ['num_rounds', 'total_timesteps']")


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
    num_envs: int,
    bszo_k: int,
) -> int:
    from problems.uhd_obj import supports_uhd_vector_objective

    if not supports_uhd_vector_objective(env_tag):
        raise ValueError("total_timesteps is currently supported only for UHD vector objective env_tag values; use num_rounds for this config.")
    eval_steps = int(steps_per_episode) * int(num_envs)
    round_steps = eval_steps * _uhd_evals_per_round(str(optimizer), bszo_k=int(bszo_k))
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
    num_envs: int,
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
            num_envs=num_envs,
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
        "target_accuracy": _optional_value(cfg, "target_accuracy", float),
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
    if "eval_episodes" in cfg:
        raise ValueError("Removed UHD config field 'eval_episodes'; use 'num_envs'.")
    num_envs = int(cfg.get("num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1 (got: {num_envs})")
    return {
        "steps_per_episode": int(cfg.get("steps_per_episode", 200)),
        "num_envs": num_envs,
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
    for name in ("num_gpus", "num_engines", "tensor_parallel_size", "sub_dataset_size"):
        _validate_optional_positive_int(fields, name)
    if fields["text_search_dim"] < 1:
        raise ValueError(f"text_search_dim must be >= 1 (got: {fields['text_search_dim']})")
    if fields["text_delta_scale"] <= 0.0:
        raise ValueError(f"text_delta_scale must be > 0 (got: {fields['text_delta_scale']})")


def _validate_optional_positive_int(fields: dict[str, Any], name: str) -> None:
    value = fields[name]
    if value is not None and value < 1:
        raise ValueError(f"{name} must be >= 1 when set (got: {value})")


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
        num_envs=fields["num_envs"],
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
