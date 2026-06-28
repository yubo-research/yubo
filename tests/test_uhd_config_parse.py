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
_parse_perturb_spec = _exp_uhd_parse._parse_perturb_spec


def test_parse_early_reject_fields_all_defaults():
    cfg = {}
    result = _parse_early_reject_fields(cfg)
    assert result.tau is None
    assert result.mode is None
    assert result.ema_beta is None
    assert result.warmup_pos is None
    assert result.quantile is None
    assert result.window is None


def test_parse_llm_sampling_rejects_multi_sample_greedy():
    with pytest.raises(ValueError, match="samples_per_prompt > 1 require temperature > 0"):
        _parse_cfg(
            {
                "env_tag": "llm:math:gsm8k",
                "policy_tag": "qwen3-1p7b-lora-r1",
                "num_rounds": 1,
                "samples_per_prompt": 2,
                "temperature": 0.0,
            }
        )


def test_parse_llm_sampling_rejects_pass_at_k_without_multiple_samples():
    with pytest.raises(ValueError, match="pass_at_k=true require samples_per_prompt > 1"):
        _parse_cfg(
            {
                "env_tag": "llm:math:gsm8k",
                "policy_tag": "qwen3-1p7b-lora-r1",
                "num_rounds": 1,
                "samples_per_prompt": 1,
                "pass_at_k": True,
            }
        )


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
    assert result.num_fit_candidates == 1
    assert result.num_fit_samples == 10
    assert result.enn_index_driver == "flat"
    assert result.sigma_range is None
    assert result.adapt_sigma is True
    assert result.acquisition == "ucb"


def test_parse_be_fields_acquisition_mu():
    result = _parse_be_fields({"be_acquisition": "mu"})
    assert result.acquisition == "mu"


def test_parse_be_fields_adapt_sigma_false():
    result = _parse_be_fields({"be_adapt_sigma": False})
    assert result.adapt_sigma is False


def test_parse_be_fields_custom_values():
    cfg = {
        "be_num_probes": 20,
        "be_num_candidates": 5,
        "be_warmup": 50,
        "be_fit_interval": 25,
        "be_enn_k": 50,
        "be_num_fit_candidates": 2,
        "be_num_fit_samples": 20,
        "be_enn_index_driver": "hnsw",
        "be_sigma_range": [1e-5, 1e-1],
    }
    result = _parse_be_fields(cfg)
    assert result.num_probes == 20
    assert result.num_candidates == 5
    assert result.warmup == 50
    assert result.fit_interval == 25
    assert result.enn_k == 50
    assert result.num_fit_candidates == 2
    assert result.num_fit_samples == 20
    assert result.enn_index_driver == "hnsw"
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
    assert result.err_ema_beta == 0.95
    assert result.max_abs_err_ema == 0.25
    assert result.min_calib_points == 10


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
        "enn_err_ema_beta": 0.9,
        "enn_max_abs_err_ema": 0.75,
        "enn_min_calib_points": 3,
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
    assert result.err_ema_beta == 0.9
    assert result.max_abs_err_ema == 0.75
    assert result.min_calib_points == 3


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


def test_parse_text_fields_bf8_storage_default_off():
    result = _parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "num_rounds": 1,
        }
    )
    assert result.bf8_storage is False


def test_parse_text_fields_bf8_storage_enabled():
    result = _parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "num_rounds": 1,
            "bf8_storage": True,
        }
    )
    assert result.bf8_storage is True


def test_parse_text_fields_semantic_update_program():
    result = _parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "num_rounds": 1,
            "llm_update_roles": ["attention_q", "mlp_down"],
            "llm_update_layer_band": "middle",
            "llm_update_expert_policy": "dense",
            "llm_update_max_targets": "4",
        }
    )

    assert result.llm_update_roles == ("attention_q", "mlp_down")
    assert result.llm_update_layer_band == "middle"
    assert result.llm_update_expert_policy == "dense"
    assert result.llm_update_max_targets == 4
