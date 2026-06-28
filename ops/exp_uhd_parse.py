from typing import Any

import click
import tomllib

from common.mapping_keys import coerce_mapping_keys, normalize_toml_key
from ops.config_overrides import parse_override_value
from ops.config_overrides import parse_overrides as _parse_overrides_raw
from ops.exp_uhd_parse_extras import apply_optional_cfg_fields, validate_llm_sampling_config
from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig
from optimizer.uhd_be_enn import parse_be_acquisition

_REQUIRED_TOML_KEYS = ("env_tag",)
_ALLOWED_OPTIMIZERS = {"simple", "simple_be", "mezo", "mezo_be", "bszo", "bszo_be"}
_OPTIONAL_TOML_KEYS = (
    "num_rounds",
    "policy_tag",
    "problem_seed",
    "noise_seed_0",
    "lr",
    "sigma",
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
    "be_num_fit_candidates",
    "be_num_fit_samples",
    "be_enn_index_driver",
    "be_sigma_range",
    "be_adapt_sigma",
    "be_acquisition",
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
    # UHD Vector/Gym settings
    "steps_per_episode",
    "num_envs",
    "deterministic_policy",
    "total_timesteps",
    "sigma",
    "seed_offset",
    "num_reps",
    # Pretrain settings
    "pretrain_search_dim",
    "pretrain_delta_scale",
    "pretrain_generation_length",
    "pretrain_rwkv_type",
    "pretrain_lora_only",
    "pretrain_basis_max_leaves",
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
    # Text settings
    "text_search_dim",
    "text_delta_scale",
    "text_basis_max_tensors",
    "text_score_mode",
    "llm_update_roles",
    "llm_update_layer_band",
    "llm_update_expert_policy",
    "llm_update_max_targets",
    # Other settings
    "bf8_storage",
    "perturb_backend",
    "eggroll_noiser",
    "eggroll_rank",
    "eggroll_group_size",
    "eggroll_freeze_nonlora",
    "use_async",
    "vllm_enforce_eager",
    "vllm_max_model_len",
    "vllm_gpu_memory_utilization",
    "vllm_max_num_seqs",
    "vllm_max_num_batched_tokens",
    "vllm_speculative_method",
    "vllm_speculative_model",
    "vllm_num_speculative_tokens",
    # Distillation settings
    "distill_teacher_model_choice",
    "distill_student_model_choice",
    "distill_dtype",
    "distill_generation_length",
    "distill_search_dim",
    "distill_delta_scale",
    "distill_lora_only",
    "distill_basis_max_leaves",
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
    "be_num_fit_candidates": 1,
    "be_num_fit_samples": 10,
    "be_enn_index_driver": "flat",
    "be_sigma_range": None,
    "be_adapt_sigma": True,
    "be_acquisition": "ucb",
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


def _parse_override_value(raw: str) -> Any:
    return parse_override_value(raw)


def _parse_overrides(override_strings: tuple[str, ...]) -> dict[str, Any]:
    return _parse_overrides_raw(
        override_strings,
        valid_keys=_ALL_TOML_KEYS,
        normalize_key=_normalize_key,
    )


def _validate_required(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_TOML_KEYS if k not in cfg]
    if "num_rounds" not in cfg and "total_timesteps" not in cfg:
        missing.extend(["num_rounds", "total_timesteps"])
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Required: {sorted(_REQUIRED_TOML_KEYS)} and one of ['num_rounds', 'total_timesteps']")


def _parse_perturb_spec(perturb: str) -> tuple[str, float | None, float | None]:
    """Parse --perturb flag into (backend, num_dim_target, num_module_target)."""
    if perturb == "eggroll":
        return "eggroll", None, None
    backend = "flat"
    if perturb == "dense":
        return backend, None, None
    if perturb.startswith("dim:"):
        return backend, float(perturb[4:]), None
    if perturb.startswith("mod:"):
        return backend, None, float(perturb[4:])
    msg = f"Invalid --perturb value: {perturb!r}. Use 'dense', 'eggroll', 'dim:<n>', or 'mod:<n>'."
    raise click.BadParameter(msg)


def _parse_perturb(perturb: str) -> tuple[float | None, float | None]:
    """Parse --perturb flag into (num_dim_target, num_module_target)."""
    _, ndt, nmt = _parse_perturb_spec(perturb)
    return ndt, nmt


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
    num_fit_candidates = int(cfg.get("be_num_fit_candidates", _BE_DEFAULTS["be_num_fit_candidates"]))
    num_fit_samples = int(cfg.get("be_num_fit_samples", _BE_DEFAULTS["be_num_fit_samples"]))
    enn_index_driver = str(cfg.get("be_enn_index_driver", _BE_DEFAULTS["be_enn_index_driver"]))
    sigma_range_raw = cfg.get("be_sigma_range", _BE_DEFAULTS["be_sigma_range"])
    sigma_range = tuple(float(v) for v in sigma_range_raw) if sigma_range_raw is not None else None
    adapt_sigma = bool(cfg.get("be_adapt_sigma", _BE_DEFAULTS["be_adapt_sigma"]))
    acquisition = parse_be_acquisition(str(cfg.get("be_acquisition", _BE_DEFAULTS["be_acquisition"])))
    return BEConfig(
        num_probes=num_probes,
        num_candidates=num_candidates,
        warmup=warmup,
        fit_interval=fit_interval,
        enn_k=enn_k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        enn_index_driver=enn_index_driver,
        sigma_range=sigma_range,
        adapt_sigma=adapt_sigma,
        acquisition=acquisition,
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


def _parse_budget_fields(cfg: dict[str, Any]) -> tuple[int, int | None]:
    from problems.uhd_obj import supports_uhd_vector_objective

    num_rounds = cfg.get("num_rounds")
    total_timesteps = cfg.get("total_timesteps")

    if num_rounds is not None:
        num_rounds = int(num_rounds)
    if total_timesteps is not None:
        total_timesteps = int(total_timesteps)

    if total_timesteps is not None and num_rounds is None:
        env_tag = str(cfg.get("env_tag", ""))
        if not supports_uhd_vector_objective(env_tag):
            raise ValueError(f"UHD vector objective (e.g. gymnax:...) required for 'total_timesteps'. Got {env_tag!r}")

        num_envs = int(cfg.get("num_envs", 1))
        steps_per_episode = int(cfg.get("steps_per_episode", 200))
        optimizer = _validate_optimizer(str(cfg.get("optimizer", "mezo")))

        if optimizer in ("bszo", "bszo_be"):
            k = int(cfg.get("bszo_k", 2))
            denom = num_envs * steps_per_episode * (k + 1)
        else:
            denom = num_envs * steps_per_episode

        if total_timesteps % denom != 0:
            raise ValueError(
                f"total_timesteps must be divisible by the derived per-round budget ({denom}) to avoid truncation; got total_timesteps={total_timesteps}."
            )
        num_rounds = total_timesteps // denom

    if num_rounds is None:
        # Should be caught by _validate_required, but for type safety:
        return 0, total_timesteps

    return num_rounds, total_timesteps


def _validate_optimizer(name: str) -> str:
    optimizer = str(name)
    if optimizer not in _ALLOWED_OPTIMIZERS:
        raise ValueError(f"Unsupported UHD optimizer {optimizer!r}. Valid optimizers: {sorted(_ALLOWED_OPTIMIZERS)}")
    return optimizer


def _parse_cfg(cfg: dict[str, Any]) -> UHDConfig:
    env_tag = str(cfg["env_tag"])
    policy_tag = cfg.get("policy_tag", None)
    if policy_tag is not None:
        policy_tag = str(policy_tag)

    num_rounds, total_timesteps = _parse_budget_fields(cfg)

    problem_seed = cfg.get("problem_seed", None)
    if problem_seed is not None:
        problem_seed = int(problem_seed)
    noise_seed_0 = cfg.get("noise_seed_0", None)
    if noise_seed_0 is not None:
        noise_seed_0 = int(noise_seed_0)

    lr = float(cfg.get("lr", 0.001))
    sigma = float(cfg.get("sigma", 0.001))
    perturb = str(cfg.get("perturb", "dim:0.5"))
    log_interval = int(cfg.get("log_interval", 1))
    accuracy_interval = int(cfg.get("accuracy_interval", 1000))
    target_accuracy = cfg.get("target_accuracy", None)
    if target_accuracy is not None:
        target_accuracy = float(target_accuracy)

    optimizer = _validate_optimizer(str(cfg.get("optimizer", "mezo")))
    batch_size = int(cfg.get("batch_size", 4096))

    bszo_k = int(cfg.get("bszo_k", 2))
    bszo_epsilon = float(cfg.get("bszo_epsilon", 1e-4))
    bszo_sigma_p_sq = float(cfg.get("bszo_sigma_p_sq", 1.0))
    bszo_sigma_e_sq = float(cfg.get("bszo_sigma_e_sq", 1.0))
    bszo_alpha = float(cfg.get("bszo_alpha", 0.1))

    early_reject = _parse_early_reject_fields(cfg)
    be = _parse_be_fields(cfg)
    enn = _parse_enn_fields(cfg)

    backend, ndt, nmt = _parse_perturb_spec(perturb)

    # Base UHDConfig fields
    config_dict = {
        "env_tag": env_tag,
        "policy_tag": policy_tag,
        "num_rounds": num_rounds,
        "problem_seed": problem_seed,
        "noise_seed_0": noise_seed_0,
        "lr": lr,
        "sigma": sigma,
        "num_dim_target": ndt,
        "num_module_target": nmt,
        "log_interval": log_interval,
        "accuracy_interval": accuracy_interval,
        "target_accuracy": target_accuracy,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "early_reject": early_reject,
        "be": be,
        "enn": enn,
        "bszo_k": bszo_k,
        "bszo_epsilon": bszo_epsilon,
        "bszo_sigma_p_sq": bszo_sigma_p_sq,
        "bszo_sigma_e_sq": bszo_sigma_e_sq,
        "bszo_alpha": bszo_alpha,
        "total_timesteps": total_timesteps,
        "perturb_backend": backend,
    }

    apply_optional_cfg_fields(config_dict, cfg)

    parsed = UHDConfig(**config_dict)
    validate_llm_sampling_config(parsed)
    return parsed
