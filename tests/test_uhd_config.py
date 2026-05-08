"""Tests for UHD configuration classes and parsing functions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ops import exp_uhd as _exp_uhd
from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig


_ALL_TOML_KEYS = _exp_uhd._ALL_TOML_KEYS
_BE_DEFAULTS = _exp_uhd._BE_DEFAULTS
_ENN_DEFAULTS = _exp_uhd._ENN_DEFAULTS
_ER_DEFAULTS = _exp_uhd._ER_DEFAULTS
_OPTIONAL_TOML_KEYS = _exp_uhd._OPTIONAL_TOML_KEYS
_REQUIRED_TOML_KEYS = _exp_uhd._REQUIRED_TOML_KEYS
_coerce_mapping_keys = _exp_uhd._coerce_mapping_keys
_load_toml_config = _exp_uhd._load_toml_config
_normalize_key = _exp_uhd._normalize_key
_parse_be_fields = _exp_uhd._parse_be_fields
_parse_cfg = _exp_uhd._parse_cfg
_parse_early_reject_fields = _exp_uhd._parse_early_reject_fields
_parse_enn_fields = _exp_uhd._parse_enn_fields
_parse_perturb = _exp_uhd._parse_perturb
_parse_perturb_spec = _exp_uhd._parse_perturb_spec
_validate_required = _exp_uhd._validate_required


class TestEarlyRejectConfig:
    """Tests for EarlyRejectConfig dataclass."""

    def test_default_creation(self):
        """Test creating EarlyRejectConfig with all None defaults."""
        cfg = EarlyRejectConfig(
            tau=None,
            mode=None,
            ema_beta=None,
            warmup_pos=None,
            quantile=None,
            window=None,
        )
        assert cfg.tau is None
        assert cfg.mode is None
        assert cfg.ema_beta is None
        assert cfg.warmup_pos is None
        assert cfg.quantile is None
        assert cfg.window is None

    def test_custom_values(self):
        """Test creating EarlyRejectConfig with custom values."""
        cfg = EarlyRejectConfig(
            tau=0.5,
            mode="ema",
            ema_beta=0.9,
            warmup_pos=100,
            quantile=0.95,
            window=50,
        )
        assert cfg.tau == 0.5
        assert cfg.mode == "ema"
        assert cfg.ema_beta == 0.9
        assert cfg.warmup_pos == 100
        assert cfg.quantile == 0.95
        assert cfg.window == 50

    def test_frozen_dataclass(self):
        """Test that EarlyRejectConfig is frozen (immutable)."""
        cfg = EarlyRejectConfig(tau=0.5, mode="test", ema_beta=0.9, warmup_pos=10, quantile=0.9, window=20)
        with pytest.raises(FrozenInstanceError):
            cfg.tau = 0.6


class TestBEConfig:
    """Tests for BEConfig dataclass."""

    def test_default_creation(self):
        """Test creating BEConfig with default values."""
        cfg = BEConfig(
            num_probes=10,
            num_candidates=10,
            warmup=20,
            fit_interval=10,
            enn_k=25,
            sigma_range=None,
        )
        assert cfg.num_probes == 10
        assert cfg.num_candidates == 10
        assert cfg.warmup == 20
        assert cfg.fit_interval == 10
        assert cfg.enn_k == 25
        assert cfg.sigma_range is None

    def test_custom_values(self):
        """Test creating BEConfig with custom values."""
        cfg = BEConfig(
            num_probes=20,
            num_candidates=5,
            warmup=50,
            fit_interval=25,
            enn_k=50,
            sigma_range=(1e-5, 1e-1),
        )
        assert cfg.num_probes == 20
        assert cfg.num_candidates == 5
        assert cfg.warmup == 50
        assert cfg.fit_interval == 25
        assert cfg.enn_k == 50
        assert cfg.sigma_range == (1e-5, 1e-1)

    def test_frozen_dataclass(self):
        """Test that BEConfig is frozen (immutable)."""
        cfg = BEConfig(num_probes=10, num_candidates=10, warmup=20, fit_interval=10, enn_k=25, sigma_range=None)
        with pytest.raises(FrozenInstanceError):
            cfg.num_probes = 15


class TestENNConfig:
    """Tests for ENNConfig dataclass."""

    def test_default_creation(self):
        """Test creating ENNConfig with default values."""
        cfg = ENNConfig(
            minus_impute=False,
            d=100,
            s=4,
            jl_seed=123,
            k=25,
            fit_interval=50,
            warmup_real_obs=200,
            refresh_interval=50,
            se_threshold=0.25,
            target="mu_minus",
            num_candidates=1,
            select_interval=1,
            embedder="direction",
            gather_t=64,
        )
        assert cfg.minus_impute is False
        assert cfg.d == 100
        assert cfg.s == 4
        assert cfg.jl_seed == 123
        assert cfg.k == 25
        assert cfg.fit_interval == 50
        assert cfg.warmup_real_obs == 200
        assert cfg.refresh_interval == 50
        assert cfg.se_threshold == 0.25
        assert cfg.target == "mu_minus"
        assert cfg.num_candidates == 1
        assert cfg.select_interval == 1
        assert cfg.embedder == "direction"
        assert cfg.gather_t == 64
        assert cfg.err_ema_beta == 0.95
        assert cfg.max_abs_err_ema == 0.25
        assert cfg.min_calib_points == 10

    def test_custom_values(self):
        """Test creating ENNConfig with custom values."""
        cfg = ENNConfig(
            minus_impute=True,
            d=200,
            s=8,
            jl_seed=456,
            k=50,
            fit_interval=100,
            warmup_real_obs=500,
            refresh_interval=100,
            se_threshold=0.5,
            target="mu_plus",
            num_candidates=5,
            select_interval=10,
            embedder="probes",
            gather_t=128,
            err_ema_beta=0.9,
            max_abs_err_ema=0.75,
            min_calib_points=3,
        )
        assert cfg.minus_impute is True
        assert cfg.d == 200
        assert cfg.s == 8
        assert cfg.jl_seed == 456
        assert cfg.k == 50
        assert cfg.fit_interval == 100
        assert cfg.warmup_real_obs == 500
        assert cfg.refresh_interval == 100
        assert cfg.se_threshold == 0.5
        assert cfg.target == "mu_plus"
        assert cfg.num_candidates == 5
        assert cfg.select_interval == 10
        assert cfg.embedder == "probes"
        assert cfg.gather_t == 128
        assert cfg.err_ema_beta == 0.9
        assert cfg.max_abs_err_ema == 0.75
        assert cfg.min_calib_points == 3

    def test_frozen_dataclass(self):
        """Test that ENNConfig is frozen (immutable)."""
        cfg = ENNConfig(
            minus_impute=False,
            d=100,
            s=4,
            jl_seed=123,
            k=25,
            fit_interval=50,
            warmup_real_obs=200,
            refresh_interval=50,
            se_threshold=0.25,
            target="mu_minus",
            num_candidates=1,
            select_interval=1,
            embedder="direction",
            gather_t=64,
        )
        with pytest.raises(FrozenInstanceError):
            cfg.d = 200


class TestUHDConfig:
    """Tests for UHDConfig dataclass."""

    def test_full_creation(self):
        """Test creating UHDConfig with all required fields."""
        early_reject = EarlyRejectConfig(tau=0.5, mode="ema", ema_beta=0.9, warmup_pos=100, quantile=0.95, window=50)
        be = BEConfig(
            num_probes=10,
            num_candidates=10,
            warmup=20,
            fit_interval=10,
            enn_k=25,
            sigma_range=None,
        )
        enn = ENNConfig(
            minus_impute=False,
            d=100,
            s=4,
            jl_seed=123,
            k=25,
            fit_interval=50,
            warmup_real_obs=200,
            refresh_interval=50,
            se_threshold=0.25,
            target="mu_minus",
            num_candidates=1,
            select_interval=1,
            embedder="direction",
            gather_t=64,
        )
        cfg = UHDConfig(
            env_tag="mnist",
            policy_tag=None,
            num_rounds=1000,
            problem_seed=42,
            noise_seed_0=123,
            lr=0.001,
            num_dim_target=0.5,
            num_module_target=None,
            log_interval=1,
            accuracy_interval=1000,
            target_accuracy=0.95,
            optimizer="mezo",
            batch_size=4096,
            early_reject=early_reject,
            be=be,
            enn=enn,
            bszo_k=2,
            bszo_epsilon=1e-4,
            bszo_sigma_p_sq=1.0,
            bszo_sigma_e_sq=1.0,
            bszo_alpha=0.1,
        )
        assert cfg.env_tag == "mnist"
        assert cfg.policy_tag is None
        assert cfg.num_rounds == 1000
        assert cfg.num_reps == 1
        assert cfg.total_timesteps is None
        assert cfg.problem_seed == 42
        assert cfg.noise_seed_0 == 123
        assert cfg.lr == 0.001
        assert cfg.num_dim_target == 0.5
        assert cfg.num_module_target is None
        assert cfg.log_interval == 1
        assert cfg.accuracy_interval == 1000
        assert cfg.target_accuracy == 0.95
        assert cfg.optimizer == "mezo"
        assert cfg.batch_size == 4096
        assert cfg.early_reject == early_reject
        assert cfg.be == be
        assert cfg.enn == enn
        assert cfg.bszo_k == 2
        assert cfg.bszo_epsilon == 1e-4
        assert cfg.bszo_sigma_p_sq == 1.0
        assert cfg.bszo_sigma_e_sq == 1.0
        assert cfg.bszo_alpha == 0.1

    def test_frozen_dataclass(self):
        """Test that UHDConfig is frozen (immutable)."""
        cfg = UHDConfig(
            env_tag="mnist",
            policy_tag=None,
            num_rounds=100,
            problem_seed=None,
            noise_seed_0=None,
            lr=0.001,
            num_dim_target=None,
            num_module_target=None,
            log_interval=1,
            accuracy_interval=100,
            target_accuracy=None,
            optimizer="mezo",
            batch_size=4096,
            early_reject=EarlyRejectConfig(None, None, None, None, None, None),
            be=BEConfig(10, 10, 20, 10, 25, None),
            enn=ENNConfig(
                minus_impute=False,
                d=100,
                s=4,
                jl_seed=123,
                k=25,
                fit_interval=50,
                warmup_real_obs=200,
                refresh_interval=50,
                se_threshold=0.25,
                target="mu_minus",
                num_candidates=1,
                select_interval=1,
                embedder="direction",
                gather_t=64,
            ),
            bszo_k=2,
            bszo_epsilon=1e-4,
            bszo_sigma_p_sq=1.0,
            bszo_sigma_e_sq=1.0,
            bszo_alpha=0.1,
        )
        with pytest.raises(FrozenInstanceError):
            cfg.env_tag = "pend"


class TestParseEarlyRejectFields:
    """Tests for _parse_early_reject_fields function."""

    def test_all_defaults(self):
        """Test parsing with all defaults (empty dict)."""
        cfg = {}
        result = _parse_early_reject_fields(cfg)
        assert result.tau is None
        assert result.mode is None
        assert result.ema_beta is None
        assert result.warmup_pos is None
        assert result.quantile is None
        assert result.window is None

    def test_custom_values(self):
        """Test parsing with custom values."""
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

    def test_partial_values(self):
        """Test parsing with some values set and others default."""
        cfg = {"er_tau": 0.3, "er_mode": "quantile"}
        result = _parse_early_reject_fields(cfg)
        assert result.tau == 0.3
        assert result.mode == "quantile"
        assert result.ema_beta is None
        assert result.warmup_pos is None
        assert result.quantile is None
        assert result.window is None

    def test_type_coercion(self):
        """Test that values are properly coerced to correct types."""
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


class TestParseBEFields:
    """Tests for _parse_be_fields function."""

    def test_all_defaults(self):
        """Test parsing with all defaults (empty dict)."""
        cfg = {}
        result = _parse_be_fields(cfg)
        assert result.num_probes == 10
        assert result.num_candidates == 10
        assert result.warmup == 20
        assert result.fit_interval == 10
        assert result.enn_k == 25
        assert result.sigma_range is None

    def test_custom_values(self):
        """Test parsing with custom values."""
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

    def test_partial_values(self):
        """Test parsing with some values set and others default."""
        cfg = {"be_num_probes": 15, "be_warmup": 30}
        result = _parse_be_fields(cfg)
        assert result.num_probes == 15
        assert result.warmup == 30
        assert result.num_candidates == 10  # default
        assert result.fit_interval == 10  # default

    def test_type_coercion(self):
        """Test that values are properly coerced to correct types."""
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


class TestParseENNFields:
    """Tests for _parse_enn_fields function."""

    def test_all_defaults(self):
        """Test parsing with all defaults (empty dict)."""
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

    def test_custom_values(self):
        """Test parsing with custom values."""
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

    def test_partial_values(self):
        """Test parsing with some values set and others default."""
        cfg = {"enn_d": 150, "enn_k": 30}
        result = _parse_enn_fields(cfg)
        assert result.d == 150
        assert result.k == 30
        assert result.s == 4  # default
        assert result.jl_seed == 123  # default

    def test_type_coercion(self):
        """Test that values are properly coerced to correct types."""
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


class TestParsePerturb:
    """Tests for _parse_perturb function."""

    def test_dense(self):
        """Test parsing 'dense' perturb."""
        ndt, nmt = _parse_perturb("dense")
        assert ndt is None
        assert nmt is None
        assert _parse_perturb_spec("dense") == ("flat", None, None)

    def test_eggroll(self):
        """Test parsing upstream EggRoll perturb materialization."""
        ndt, nmt = _parse_perturb("eggroll")
        assert ndt is None
        assert nmt is None
        assert _parse_perturb_spec("eggroll") == ("eggroll", None, None)

    def test_dim(self):
        """Test parsing 'dim:<n>' perturb."""
        ndt, nmt = _parse_perturb("dim:0.5")
        assert ndt == 0.5
        assert nmt is None

    def test_mod(self):
        """Test parsing 'mod:<n>' perturb."""
        ndt, nmt = _parse_perturb("mod:0.3")
        assert ndt is None
        assert nmt == 0.3

    def test_invalid_value(self):
        """Test that invalid perturb values raise BadParameter."""
        from click import BadParameter

        with pytest.raises(BadParameter):
            _parse_perturb("invalid")


class TestParseCfg:
    """Tests for _parse_cfg function."""

    def test_minimal_config(self):
        """Test parsing with minimal required config."""
        cfg = {
            "env_tag": "mnist",
            "num_rounds": 100,
        }
        result = _parse_cfg(cfg)
        assert result.env_tag == "mnist"
        assert result.policy_tag is None
        assert result.num_rounds == 100
        assert result.num_reps == 1
        assert result.total_timesteps is None
        assert result.problem_seed is None
        assert result.noise_seed_0 is None
        assert result.lr == 0.001  # default
        assert result.perturb_backend == "flat"
        assert result.num_dim_target == 0.5  # default from "dim:0.5"
        assert result.num_module_target is None
        assert result.optimizer == "mezo"  # default

    def test_full_config(self):
        """Test parsing with all config options."""
        cfg = {
            "env_tag": "pend",
            "policy_tag": "some-policy",
            "num_rounds": 500,
            "num_reps": 30,
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
        assert result.num_reps == 30
        assert result.total_timesteps is None
        assert result.problem_seed == 42
        assert result.noise_seed_0 == 123
        assert result.lr == 0.01
        assert result.perturb_backend == "flat"
        assert result.num_dim_target is None
        assert result.num_module_target == 0.3
        assert result.log_interval == 10
        assert result.accuracy_interval == 500
        assert result.target_accuracy == 0.99
        assert result.optimizer == "simple"
        assert result.batch_size == 2048
        # Check nested configs
        assert result.early_reject.tau == 0.5
        assert result.early_reject.mode == "ema"
        assert result.be.num_probes == 20
        assert result.enn.d == 200
        assert result.bszo_k == 4

    def test_eggroll_perturb_config(self):
        """Test parsing perturb='eggroll' with noiser materialization options."""
        result = _parse_cfg(
            {
                "env_tag": "gymnax:CartPole-v1",
                "policy_tag": "eggroll-ac-mlp-8x1-pqn",
                "num_rounds": 10,
                "perturb": "eggroll",
                "eggroll_noiser": "eggroll",
                "eggroll_rank": 2,
                "eggroll_group_size": 4,
                "eggroll_freeze_nonlora": True,
            }
        )
        assert result.perturb_backend == "eggroll"
        assert result.num_dim_target is None
        assert result.num_module_target is None
        assert result.eggroll_noiser == "eggroll"
        assert result.eggroll_rank == 2
        assert result.eggroll_group_size == 4
        assert result.eggroll_freeze_nonlora is True

    def test_none_seeds_stay_none(self):
        """Test that None seeds stay None (not converted to int)."""
        cfg = {
            "env_tag": "mnist",
            "num_rounds": 100,
            "problem_seed": None,
            "noise_seed_0": None,
        }
        result = _parse_cfg(cfg)
        assert result.problem_seed is None
        assert result.noise_seed_0 is None

    def test_optional_target_accuracy_none(self):
        """Test that target_accuracy can be None."""
        cfg = {
            "env_tag": "mnist",
            "num_rounds": 100,
            "target_accuracy": None,
        }
        result = _parse_cfg(cfg)
        assert result.target_accuracy is None

    def test_total_timesteps_derives_num_rounds_for_eggroll_mezo(self):
        """Test deriving UHD eval rounds from an EggRoll total timestep budget."""
        cfg = {
            "env_tag": "gymnax:CartPole-v1",
            "total_timesteps": 500_000_000,
            "steps_per_episode": 500,
            "eval_episodes": 8,
            "optimizer": "mezo",
        }
        result = _parse_cfg(cfg)
        assert result.total_timesteps == 500_000_000
        assert result.num_rounds == 125_000

    def test_total_timesteps_derives_num_rounds_for_eggroll_bszo(self):
        """Test BSZO total timestep derivation accounts for base plus directions."""
        cfg = {
            "env_tag": "gymnax:CartPole-v1",
            "total_timesteps": 1_000,
            "steps_per_episode": 10,
            "eval_episodes": 5,
            "optimizer": "bszo",
            "bszo_k": 4,
        }
        result = _parse_cfg(cfg)
        assert result.num_rounds == 4

    def test_total_timesteps_derives_num_rounds_for_pretrain_vector_objective(self):
        """Test UHD accepts pretraining tags through the vector objective path."""
        cfg = {
            "env_tag": "pretrain:hyperscalees:gsm8k-7w3b",
            "policy_tag": "hyperscalees-rwkv-7w3b-lora-r1",
            "total_timesteps": 819_200,
            "steps_per_episode": 1,
            "eval_episodes": 1,
            "optimizer": "mezo_be",
        }
        result = _parse_cfg(cfg)
        assert result.num_rounds == 819_200
        assert result.policy_tag == "hyperscalees-rwkv-7w3b-lora-r1"
        assert result.pretrain_lora_only is True
        assert result.pretrain_basis_max_leaves == 32

    def test_pretrain_hyperscalees_overrides_parse(self):
        """Test real HyperscaleES UHD-only knobs parse and validate."""
        cfg = {
            "env_tag": "pretrain:hyperscalees:gsm8k-7w3b",
            "policy_tag": "hyperscalees-rwkv-7w3b-lora-r1",
            "num_rounds": 1,
            "pretrain_search_dim": 128,
            "pretrain_delta_scale": 0.01,
            "pretrain_generation_length": 64,
            "pretrain_rwkv_type": "ScanRWKV",
            "pretrain_lora_only": False,
            "pretrain_basis_max_leaves": 0,
        }
        result = _parse_cfg(cfg)
        assert result.pretrain_search_dim == 128
        assert result.pretrain_delta_scale == 0.01
        assert result.pretrain_generation_length == 64
        assert result.pretrain_rwkv_type == "ScanRWKV"
        assert result.pretrain_lora_only is False
        assert result.pretrain_basis_max_leaves is None

    def test_total_timesteps_only_rejects_non_eggroll_env(self):
        """Test total_timesteps is not accepted when UHD cannot infer env-step cost."""
        cfg = {
            "env_tag": "mnist",
            "total_timesteps": 1_000,
        }
        with pytest.raises(ValueError, match="UHD vector objective"):
            _parse_cfg(cfg)


class TestNormalizeKey:
    """Tests for _normalize_key function."""

    def test_no_hyphen(self):
        """Test key without hyphen."""
        assert _normalize_key("env_tag") == "env_tag"

    def test_with_hyphen(self):
        """Test key with hyphen gets converted to underscore."""
        assert _normalize_key("env-tag") == "env_tag"
        assert _normalize_key("policy-tag") == "policy_tag"
        assert _normalize_key("num-rounds") == "num_rounds"
        assert _normalize_key("problem-seed") == "problem_seed"

    def test_multiple_hyphens(self):
        """Test key with multiple hyphens."""
        assert _normalize_key("early-reject-tau") == "early_reject_tau"


class TestCoerceMappingKeys:
    """Tests for _coerce_mapping_keys function."""

    def test_valid_keys(self):
        """Test coercing valid keys."""
        raw = {"env_tag": "mnist", "num_rounds": 100}
        result = _coerce_mapping_keys(raw, source="test")
        assert result == {"env_tag": "mnist", "num_rounds": 100}

    def test_hyphenated_keys(self):
        """Test coercing hyphenated keys to underscored."""
        raw = {"env-tag": "mnist", "num-rounds": 100}
        result = _coerce_mapping_keys(raw, source="test")
        assert result == {"env_tag": "mnist", "num_rounds": 100}

    def test_invalid_key_raises(self):
        """Test that invalid keys raise ValueError."""
        raw = {"invalid_key": "value"}
        with pytest.raises(ValueError) as exc_info:
            _coerce_mapping_keys(raw, source="test_source")
        assert "invalid_key" in str(exc_info.value)
        assert "test_source" in str(exc_info.value)

    def test_non_dict_raises(self):
        """Test that non-dict input raises TypeError."""
        with pytest.raises(TypeError):
            _coerce_mapping_keys("not_a_dict", source="test")


class TestValidateRequired:
    """Tests for _validate_required function."""

    def test_all_required_present(self):
        """Test validation passes with all required keys."""
        cfg = {"env_tag": "mnist", "num_rounds": 100}
        _validate_required(cfg)  # Should not raise

    def test_missing_required_raises(self):
        """Test validation raises when budget keys are missing."""
        cfg = {"env_tag": "mnist"}  # missing num_rounds or total_timesteps
        with pytest.raises(ValueError) as exc_info:
            _validate_required(cfg)
        assert "num_rounds" in str(exc_info.value)
        assert "total_timesteps" in str(exc_info.value)

    def test_missing_multiple_required_raises(self):
        """Test validation raises when multiple required keys are missing."""
        cfg = {}  # missing both
        with pytest.raises(ValueError) as exc_info:
            _validate_required(cfg)
        assert "env_tag" in str(exc_info.value)

    def test_total_timesteps_budget_present(self):
        """Test validation accepts total_timesteps instead of num_rounds."""
        cfg = {"env_tag": "gymnax:CartPole-v1", "total_timesteps": 100}
        _validate_required(cfg)


class TestLoadTomlConfig:
    """Tests for _load_toml_config function."""

    def test_root_level_config(self, tmp_path):
        """Test loading TOML config from root level."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('env_tag = "mnist"\nnum_rounds = 100\n')
        result = _load_toml_config(str(cfg_file))
        assert result["env_tag"] == "mnist"
        assert result["num_rounds"] == 100


def test_exp_uhd_local_uses_config_num_reps(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from ops.exp_uhd import cli

    called = {}

    def fake_batch_local(cfg, num_reps, results_dir, workers):
        called.update(cfg=cfg, num_reps=num_reps, results_dir=results_dir, workers=workers)

    monkeypatch.setattr("ops.uhd_batch._batch_local", fake_batch_local)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('[uhd]\nenv_tag = "f:sphere-2d"\nnum_rounds = 1\nnum_reps = 30\n')

    result = CliRunner().invoke(cli, ["local", str(cfg_file), "--workers", "2", "--results-dir", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert called["num_reps"] == 30
    assert called["workers"] == 2


def test_exp_uhd_local_batches_total_timesteps_budget(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from ops.exp_uhd import cli

    called = {}

    def fake_batch_local(cfg, num_reps, results_dir, workers):
        called.update(cfg=cfg, num_reps=num_reps, results_dir=results_dir, workers=workers)

    monkeypatch.setattr("ops.uhd_batch._batch_local", fake_batch_local)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        """
[uhd]
env_tag = "gymnax:CartPole-v1"
policy_tag = "eggroll-ac-mlp-8x1-pqn"
total_timesteps = 8000
num_reps = 2
steps_per_episode = 500
eval_episodes = 8
""".lstrip()
    )

    result = CliRunner().invoke(cli, ["local", str(cfg_file)])

    assert result.exit_code == 0, result.output
    assert called["num_reps"] == 2
    assert called["cfg"]["total_timesteps"] == 8000
    assert called["cfg"]["num_rounds"] == 2

    def test_uhd_section_config(self, tmp_path):
        """Test loading TOML config from [uhd] section."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 100\n')
        result = _load_toml_config(str(cfg_file))
        assert result["env_tag"] == "mnist"
        assert result["num_rounds"] == 100

    def test_hyphenated_keys_normalized(self, tmp_path):
        """Test that hyphenated keys in TOML get normalized."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('env-tag = "mnist"\nnum-rounds = 100\n')
        result = _load_toml_config(str(cfg_file))
        assert result["env_tag"] == "mnist"
        assert result["num_rounds"] == 100

    def test_invalid_toml_raises(self, tmp_path):
        """Test that invalid TOML raises TOMLDecodeError."""
        import tomllib

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("invalid toml content [[[")
        with pytest.raises(tomllib.TOMLDecodeError):
            _load_toml_config(str(cfg_file))

    def test_missing_file_raises(self, tmp_path):
        """Test that missing file raises OSError."""
        with pytest.raises(OSError):
            _load_toml_config(str(tmp_path / "nonexistent.toml"))


class TestDefaultConstants:
    """Tests for default constants."""

    def test_er_defaults(self):
        """Test ER defaults are all None."""
        for key, value in _ER_DEFAULTS.items():
            assert key.startswith("er_")
            assert value is None

    def test_be_defaults_types(self):
        """Test BE defaults have correct types."""
        assert isinstance(_BE_DEFAULTS["be_num_probes"], int)
        assert isinstance(_BE_DEFAULTS["be_num_candidates"], int)
        assert isinstance(_BE_DEFAULTS["be_warmup"], int)
        assert isinstance(_BE_DEFAULTS["be_fit_interval"], int)
        assert isinstance(_BE_DEFAULTS["be_enn_k"], int)
        assert _BE_DEFAULTS["be_sigma_range"] is None

    def test_enn_defaults_types(self):
        """Test ENN defaults have correct types."""
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

    def test_required_keys(self):
        """Test required keys are correct."""
        assert _REQUIRED_TOML_KEYS == ("env_tag",)

    def test_all_toml_keys_completeness(self):
        """Test that _ALL_TOML_KEYS contains all required and optional keys."""
        for key in _REQUIRED_TOML_KEYS:
            assert key in _ALL_TOML_KEYS
        for key in _OPTIONAL_TOML_KEYS:
            assert key in _ALL_TOML_KEYS
