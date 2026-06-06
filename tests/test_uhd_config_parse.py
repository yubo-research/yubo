"""Tests for UHD TOML field parsing helpers (_parse_*)."""

from __future__ import annotations

import pytest

import ops.exp_uhd_parse as _exp_uhd_parse

_parse_be_fields = _exp_uhd_parse._parse_be_fields
_parse_cfg = _exp_uhd_parse._parse_cfg
_validate_required = _exp_uhd_parse._validate_required
_parse_early_reject_fields = _exp_uhd_parse._parse_early_reject_fields
_parse_enn_fields = _exp_uhd_parse._parse_enn_fields
_parse_perturb = _exp_uhd_parse._parse_perturb


def test_parse_early_reject_fields_all_defaults():
    cfg = {}
    result = _parse_early_reject_fields(cfg)
    assert result.tau is None
    assert result.mode is None
    assert result.ema_beta is None
    assert result.warmup_pos is None
    assert result.quantile is None
    assert result.window is None


def test_parse_early_reject_fields_custom_values():
    cfg = {
        "er_tau": 0.5,
        "er_mode": "ema",
        "er_ema_beta": 0.9,
        "er_warmup_pos": 100,
        "er_quantile": 0.95,
        "er_window": 50,
    }
    result = _parse_early_reject_fields(cfg)
    assert result.tau == 0.5
    assert result.mode == "ema"
    assert result.ema_beta == 0.9
    assert result.warmup_pos == 100
    assert result.quantile == 0.95
    assert result.window == 50


def test_parse_early_reject_fields_partial_values():
    cfg = {"er_tau": 0.3, "er_mode": "quantile"}
    result = _parse_early_reject_fields(cfg)
    assert result.tau == 0.3
    assert result.mode == "quantile"
    assert result.ema_beta is None
    assert result.warmup_pos is None
    assert result.quantile is None
    assert result.window is None


def test_parse_early_reject_fields_type_coercion():
    cfg = {
        "er_tau": "0.5",
        "er_ema_beta": "0.9",
        "er_warmup_pos": "100",
        "er_quantile": "0.95",
        "er_window": "50",
    }
    result = _parse_early_reject_fields(cfg)
    assert result.tau == 0.5
    assert result.ema_beta == 0.9
    assert result.warmup_pos == 100
    assert result.quantile == 0.95
    assert result.window == 50


def test_parse_be_fields_all_defaults():
    cfg = {}
    result = _parse_be_fields(cfg)
    assert result.num_probes == 10
    assert result.num_candidates == 10
    assert result.warmup == 20
    assert result.fit_interval == 10
    assert result.enn_k == 25
    assert result.sigma_range is None


def test_parse_be_fields_custom_values():
    cfg = {
        "be_num_probes": 20,
        "be_num_candidates": 5,
        "be_warmup": 50,
        "be_fit_interval": 25,
        "be_enn_k": 50,
        "be_sigma_range": [1e-5, 1e-1],
    }
    result = _parse_be_fields(cfg)
    assert result.num_probes == 20
    assert result.num_candidates == 5
    assert result.warmup == 50
    assert result.fit_interval == 25
    assert result.enn_k == 50
    assert result.sigma_range == (1e-5, 1e-1)


def test_parse_be_fields_partial_values():
    cfg = {"be_num_probes": 15, "be_warmup": 30}
    result = _parse_be_fields(cfg)
    assert result.num_probes == 15
    assert result.warmup == 30
    assert result.num_candidates == 10
    assert result.fit_interval == 10


def test_parse_be_fields_type_coercion():
    cfg = {
        "be_num_probes": "20",
        "be_num_candidates": "5",
        "be_warmup": "50",
        "be_fit_interval": "25",
        "be_enn_k": "50",
    }
    result = _parse_be_fields(cfg)
    assert result.num_probes == 20
    assert result.num_candidates == 5
    assert result.warmup == 50
    assert result.fit_interval == 25
    assert result.enn_k == 50


def test_parse_enn_fields_all_defaults():
    cfg = {}
    result = _parse_enn_fields(cfg)
    assert result.minus_impute is False
    assert result.d == 100
    assert result.s == 4
    assert result.jl_seed == 123
    assert result.k == 25
    assert result.fit_interval == 50
    assert result.warmup_real_obs == 200
    assert result.refresh_interval == 50
    assert result.se_threshold == 0.25
    assert result.target == "mu_minus"
    assert result.num_candidates == 1
    assert result.select_interval == 1
    assert result.embedder == "direction"
    assert result.gather_t == 64


def test_parse_enn_fields_custom_values():
    cfg = {
        "enn_minus_impute": True,
        "enn_d": 200,
        "enn_s": 8,
        "enn_jl_seed": 456,
        "enn_k": 50,
        "enn_fit_interval": 100,
        "enn_warmup_real_obs": 500,
        "enn_refresh_interval": 100,
        "enn_se_threshold": 0.5,
        "enn_target": "mu_plus",
        "enn_num_candidates": 5,
        "enn_select_interval": 10,
        "enn_embedder": "probes",
        "enn_gather_t": 128,
    }
    result = _parse_enn_fields(cfg)
    assert result.minus_impute is True
    assert result.d == 200
    assert result.s == 8
    assert result.jl_seed == 456
    assert result.k == 50
    assert result.fit_interval == 100
    assert result.warmup_real_obs == 500
    assert result.refresh_interval == 100
    assert result.se_threshold == 0.5
    assert result.target == "mu_plus"
    assert result.num_candidates == 5
    assert result.select_interval == 10
    assert result.embedder == "probes"
    assert result.gather_t == 128


def test_parse_enn_fields_partial_values():
    cfg = {"enn_d": 150, "enn_k": 30}
    result = _parse_enn_fields(cfg)
    assert result.d == 150
    assert result.k == 30
    assert result.s == 4
    assert result.jl_seed == 123


def test_parse_enn_fields_type_coercion():
    cfg = {
        "enn_minus_impute": "True",
        "enn_d": "200",
        "enn_s": "8",
        "enn_jl_seed": "456",
        "enn_se_threshold": "0.5",
        "enn_num_candidates": "5",
    }
    result = _parse_enn_fields(cfg)
    assert result.minus_impute is True
    assert result.d == 200
    assert result.s == 8
    assert result.jl_seed == 456
    assert result.se_threshold == 0.5
    assert result.num_candidates == 5


def test_parse_perturb_dense():
    ndt, nmt = _parse_perturb("dense")
    assert ndt is None
    assert nmt is None


def test_parse_perturb_dim():
    ndt, nmt = _parse_perturb("dim:0.5")
    assert ndt == 0.5
    assert nmt is None


def test_parse_perturb_mod():
    ndt, nmt = _parse_perturb("mod:0.3")
    assert ndt is None
    assert nmt == 0.3


def test_parse_perturb_invalid_value():
    from click import BadParameter

    with pytest.raises(BadParameter):
        _parse_perturb("invalid")


def test_validate_required_raises_without_policy_tag():
    cfg = {
        "env_tag": "mnist",
        "num_rounds": 100,
    }
    with pytest.raises(ValueError, match="policy_tag"):
        _validate_required(cfg)


def test_parse_cfg_minimal_config():
    cfg = {
        "env_tag": "mnist",
        "policy_tag": "pure-function",
        "num_rounds": 100,
    }
    result = _parse_cfg(cfg)
    assert result.env_tag == "mnist"
    assert result.policy_tag == "pure-function"
    assert result.num_rounds == 100
    assert result.problem_seed is None
    assert result.noise_seed_0 is None
    assert result.lr == 0.001
    assert result.num_dim_target == 0.5
    assert result.num_module_target is None
    assert result.optimizer == "mezo"


def test_parse_cfg_full_config():
    cfg = {
        "env_tag": "pend",
        "policy_tag": "some-policy",
        "num_rounds": 500,
        "problem_seed": 42,
        "noise_seed_0": 123,
        "lr": 0.01,
        "perturb": "mod:0.3",
        "log_interval": 10,
        "accuracy_interval": 500,
        "target_accuracy": 0.99,
        "optimizer": "simple",
        "batch_size": 2048,
        "er_tau": 0.5,
        "er_mode": "ema",
        "be_num_probes": 20,
        "enn_d": 200,
        "bszo_k": 4,
    }
    result = _parse_cfg(cfg)
    assert result.env_tag == "pend"
    assert result.policy_tag == "some-policy"
    assert result.num_rounds == 500
    assert result.problem_seed == 42
    assert result.noise_seed_0 == 123
    assert result.lr == 0.01
    assert result.num_dim_target is None
    assert result.num_module_target == 0.3
    assert result.log_interval == 10
    assert result.accuracy_interval == 500
    assert result.target_accuracy == 0.99
    assert result.optimizer == "simple"
    assert result.batch_size == 2048
    assert result.early_reject.tau == 0.5
    assert result.early_reject.mode == "ema"
    assert result.be.num_probes == 20
    assert result.enn.d == 200
    assert result.bszo_k == 4


def test_parse_cfg_none_seeds_stay_none():
    cfg = {
        "env_tag": "mnist",
        "policy_tag": "pure-function",
        "num_rounds": 100,
        "problem_seed": None,
        "noise_seed_0": None,
    }
    result = _parse_cfg(cfg)
    assert result.problem_seed is None
    assert result.noise_seed_0 is None


def test_parse_cfg_optional_target_accuracy_none():
    cfg = {
        "env_tag": "mnist",
        "policy_tag": "pure-function",
        "num_rounds": 100,
        "target_accuracy": None,
    }
    result = _parse_cfg(cfg)
    assert result.target_accuracy is None
