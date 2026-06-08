"""Tests for UHD TOML loading, key normalization, allowlists, and defaults."""

from __future__ import annotations

import pytest

import ops.exp_uhd_parse as _exp_uhd_parse
from common.mapping_keys import coerce_mapping_keys, normalize_toml_key

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


def test_mapping_keys_module_normalize_toml_key():
    assert normalize_toml_key("num-arms") == "num_arms"


def test_mapping_keys_module_coerce_mapping_keys():
    out = coerce_mapping_keys(
        {"num-arms": 1},
        source="test",
        valid_keys={"num_arms"},
        not_mapping_msg="expected dict",
    )
    assert out["num_arms"] == 1


def _capture_batch_local(monkeypatch):
    called = {}

    def fake_batch_local(cfg, num_reps, results_dir, workers):
        called.update(cfg=cfg, num_reps=num_reps, results_dir=results_dir, workers=workers)

    monkeypatch.setattr("ops.uhd_batch._batch_local", fake_batch_local)
    return called


def _write_cartpole_budget_config(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        """
[uhd]
env_tag = "gymnax:CartPole-v1"
policy_tag = "eggroll-ac-mlp-8x1-pqn"
total_timesteps = 8000
num_reps = 2
steps_per_episode = 500
num_envs = 8
""".lstrip()
    )
    return cfg_file


def _invoke_cartpole_local(monkeypatch, tmp_path, extra_args):
    from click.testing import CliRunner

    from ops.exp_uhd import cli

    called = _capture_batch_local(monkeypatch)
    cfg_file = _write_cartpole_budget_config(tmp_path)
    result = CliRunner().invoke(cli, ["local", str(cfg_file), *extra_args])
    return result, called


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
    assert "total_timesteps" in str(exc_info.value)


def test_validate_required_missing_multiple_required_raises():
    cfg = {}
    with pytest.raises(ValueError) as exc_info:
        _validate_required(cfg)
    assert "env_tag" in str(exc_info.value)


def test_validate_required_accepts_total_timesteps_budget():
    cfg = {"env_tag": "gymnax:CartPole-v1", "total_timesteps": 100}
    _validate_required(cfg)


def _assert_mnist_rounds_toml(tmp_path, body: str):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(body)
    result = _load_toml_config(str(cfg_file))
    assert result["env_tag"] == "mnist"
    assert result["num_rounds"] == 100


def test_load_toml_config_root_level_config(tmp_path):
    _assert_mnist_rounds_toml(tmp_path, 'env_tag = "mnist"\nnum_rounds = 100\n')


def test_load_toml_config_uhd_section_config(tmp_path):
    _assert_mnist_rounds_toml(tmp_path, '[uhd]\nenv_tag = "mnist"\nnum_rounds = 100\n')


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
    assert isinstance(_ENN_DEFAULTS["enn_err_ema_beta"], float)
    assert isinstance(_ENN_DEFAULTS["enn_max_abs_err_ema"], float)
    assert isinstance(_ENN_DEFAULTS["enn_min_calib_points"], int)


def test_default_constants_required_keys():
    assert _REQUIRED_TOML_KEYS == ("env_tag",)


def test_default_constants_all_toml_keys_completeness():
    for key in _REQUIRED_TOML_KEYS:
        assert key in _ALL_TOML_KEYS
    for key in _OPTIONAL_TOML_KEYS:
        assert key in _ALL_TOML_KEYS


def test_default_constants_er_default_keys_in_allowlist():
    for key in _ER_DEFAULTS:
        assert key in _ALL_TOML_KEYS, key


def test_exp_uhd_local_uses_config_num_reps(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from ops.exp_uhd import cli

    called = {}

    def fake_batch_local(cfg, num_reps, results_dir, workers):
        called.update(cfg=cfg, num_reps=num_reps, results_dir=results_dir, workers=workers)

    monkeypatch.setattr("ops.uhd_batch._batch_local", fake_batch_local)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('[uhd]\nenv_tag = "f:sphere-2d"\nnum_rounds = 1\nnum_reps = 30\n')

    result = CliRunner().invoke(
        cli,
        [
            "local",
            str(cfg_file),
            "--workers",
            "2",
            "--results-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["num_reps"] == 30
    assert called["workers"] == 2


def test_exp_uhd_local_batches_total_timesteps_budget(monkeypatch, tmp_path):
    result, called = _invoke_cartpole_local(monkeypatch, tmp_path, [])

    assert result.exit_code == 0, result.output
    assert called["num_reps"] == 2
    assert called["cfg"]["total_timesteps"] == 8000
    assert called["cfg"]["num_rounds"] == 2


def test_exp_uhd_local_applies_cli_overrides(monkeypatch, tmp_path):
    result, called = _invoke_cartpole_local(
        monkeypatch,
        tmp_path,
        ["--opt", "total_timesteps=4000", "--opt", "num_reps=3"],
    )

    assert result.exit_code == 0, result.output
    assert called["num_reps"] == 3
    assert called["cfg"]["total_timesteps"] == 4000
    assert called["cfg"]["num_rounds"] == 1
