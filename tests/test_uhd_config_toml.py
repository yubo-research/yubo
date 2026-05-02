"""Tests for UHD TOML loading, key normalization, allowlists, and defaults."""

from __future__ import annotations

import pytest

import ops.exp_uhd_parse as _exp_uhd_parse

_ALL_TOML_KEYS = _exp_uhd_parse._ALL_TOML_KEYS
_BE_DEFAULTS = _exp_uhd_parse._BE_DEFAULTS
_ENN_DEFAULTS = _exp_uhd_parse._ENN_DEFAULTS
_ER_DEFAULTS = _exp_uhd_parse._ER_DEFAULTS
_OPTIONAL_TOML_KEYS = _exp_uhd_parse._OPTIONAL_TOML_KEYS
_REQUIRED_TOML_KEYS = _exp_uhd_parse._REQUIRED_TOML_KEYS
_coerce_mapping_keys = _exp_uhd_parse._coerce_mapping_keys
_load_toml_config = _exp_uhd_parse._load_toml_config
_normalize_key = _exp_uhd_parse._normalize_key
_parse_cfg = _exp_uhd_parse._parse_cfg
_validate_required = _exp_uhd_parse._validate_required


def test_normalize_key_no_hyphen():
    assert _normalize_key("env_tag") == "env_tag"


def test_normalize_key_with_hyphen():
    assert _normalize_key("env-tag") == "env_tag"
    assert _normalize_key("policy-tag") == "policy_tag"
    assert _normalize_key("num-rounds") == "num_rounds"
    assert _normalize_key("problem-seed") == "problem_seed"


def test_normalize_key_multiple_hyphens():
    assert _normalize_key("early-reject-tau") == "early_reject_tau"


def test_coerce_mapping_keys_valid_keys():
    raw = {"env_tag": "mnist", "num_rounds": 100}
    result = _coerce_mapping_keys(raw, source="test")
    assert result == {"env_tag": "mnist", "num_rounds": 100}


def test_coerce_mapping_keys_hyphenated_keys():
    raw = {"env-tag": "mnist", "num-rounds": 100}
    result = _coerce_mapping_keys(raw, source="test")
    assert result == {"env_tag": "mnist", "num_rounds": 100}


def test_coerce_mapping_keys_invalid_key_raises():
    raw = {"invalid_key": "value"}
    with pytest.raises(ValueError) as exc_info:
        _coerce_mapping_keys(raw, source="test_source")
    assert "invalid_key" in str(exc_info.value)
    assert "test_source" in str(exc_info.value)


def test_coerce_mapping_keys_non_dict_raises():
    with pytest.raises(TypeError):
        _coerce_mapping_keys("not_a_dict", source="test")


def test_validate_required_all_required_present():
    cfg = {"env_tag": "mnist", "num_rounds": 100}
    _validate_required(cfg)


def test_validate_required_missing_required_raises():
    cfg = {"env_tag": "mnist"}
    with pytest.raises(ValueError) as exc_info:
        _validate_required(cfg)
    assert "num_rounds" in str(exc_info.value)


def test_validate_required_missing_multiple_required_raises():
    cfg = {}
    with pytest.raises(ValueError) as exc_info:
        _validate_required(cfg)
    assert "env_tag" in str(exc_info.value)
    assert "num_rounds" in str(exc_info.value)


def test_load_toml_config_root_level_config(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('env_tag = "mnist"\nnum_rounds = 100\n')
    result = _load_toml_config(str(cfg_file))
    assert result["env_tag"] == "mnist"
    assert result["num_rounds"] == 100


def test_load_toml_config_uhd_section_config(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 100\n')
    result = _load_toml_config(str(cfg_file))
    assert result["env_tag"] == "mnist"
    assert result["num_rounds"] == 100


def test_load_toml_config_hyphenated_keys_normalized(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('env-tag = "mnist"\nnum-rounds = 100\n')
    result = _load_toml_config(str(cfg_file))
    assert result["env_tag"] == "mnist"
    assert result["num_rounds"] == 100


def test_load_toml_config_invalid_toml_raises(tmp_path):
    import tomllib

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("invalid toml content [[[")
    with pytest.raises(tomllib.TOMLDecodeError):
        _load_toml_config(str(cfg_file))


def test_load_toml_config_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        _load_toml_config(str(tmp_path / "nonexistent.toml"))


def test_load_toml_config_er_keys_survive_toml_load(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\ner_tau = 0.1\ner_mode = "ema"\ner-ema-beta = 0.9\n')
    raw = _load_toml_config(str(cfg_file))
    parsed = _parse_cfg(raw)
    assert parsed.early_reject.tau == 0.1
    assert parsed.early_reject.mode == "ema"
    assert parsed.early_reject.ema_beta == 0.9


def test_default_constants_er_defaults():
    for key, value in _ER_DEFAULTS.items():
        assert key.startswith("er_")
        assert value is None


def test_default_constants_be_defaults_types():
    assert isinstance(_BE_DEFAULTS["be_num_probes"], int)
    assert isinstance(_BE_DEFAULTS["be_num_candidates"], int)
    assert isinstance(_BE_DEFAULTS["be_warmup"], int)
    assert isinstance(_BE_DEFAULTS["be_fit_interval"], int)
    assert isinstance(_BE_DEFAULTS["be_enn_k"], int)
    assert _BE_DEFAULTS["be_sigma_range"] is None


def test_default_constants_enn_defaults_types():
    assert isinstance(_ENN_DEFAULTS["enn_minus_impute"], bool)
    assert isinstance(_ENN_DEFAULTS["enn_d"], int)
    assert isinstance(_ENN_DEFAULTS["enn_s"], int)
    assert isinstance(_ENN_DEFAULTS["enn_jl_seed"], int)
    assert isinstance(_ENN_DEFAULTS["enn_k"], int)
    assert isinstance(_ENN_DEFAULTS["enn_fit_interval"], int)
    assert isinstance(_ENN_DEFAULTS["enn_warmup_real_obs"], int)
    assert isinstance(_ENN_DEFAULTS["enn_refresh_interval"], int)
    assert isinstance(_ENN_DEFAULTS["enn_se_threshold"], float)
    assert isinstance(_ENN_DEFAULTS["enn_target"], str)
    assert isinstance(_ENN_DEFAULTS["enn_num_candidates"], int)
    assert isinstance(_ENN_DEFAULTS["enn_select_interval"], int)
    assert isinstance(_ENN_DEFAULTS["enn_embedder"], str)
    assert isinstance(_ENN_DEFAULTS["enn_gather_t"], int)


def test_default_constants_required_keys():
    assert _REQUIRED_TOML_KEYS == ("env_tag", "num_rounds")


def test_default_constants_all_toml_keys_completeness():
    for key in _REQUIRED_TOML_KEYS:
        assert key in _ALL_TOML_KEYS
    for key in _OPTIONAL_TOML_KEYS:
        assert key in _ALL_TOML_KEYS


def test_default_constants_er_default_keys_in_allowlist():
    for key in _ER_DEFAULTS:
        assert key in _ALL_TOML_KEYS, key
