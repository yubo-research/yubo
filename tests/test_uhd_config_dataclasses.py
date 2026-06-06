"""UHD dataclass (EarlyReject, BE, ENN, UHD) construction and immutability tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig


def test_early_reject_config_default_creation():
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


def test_early_reject_config_custom_values():
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


def test_early_reject_config_frozen_dataclass():
    cfg = EarlyRejectConfig(tau=0.5, mode="test", ema_beta=0.9, warmup_pos=10, quantile=0.9, window=20)
    with pytest.raises(FrozenInstanceError):
        cfg.tau = 0.6


def _assert_be_config_fields(
    cfg,
    *,
    num_probes,
    num_candidates,
    warmup,
    fit_interval,
    enn_k,
    sigma_range,
):
    assert cfg.num_probes == num_probes
    assert cfg.num_candidates == num_candidates
    assert cfg.warmup == warmup
    assert cfg.fit_interval == fit_interval
    assert cfg.enn_k == enn_k
    assert cfg.sigma_range == sigma_range


def test_be_config_default_creation():
    cfg = BEConfig(
        num_probes=10,
        num_candidates=10,
        warmup=20,
        fit_interval=10,
        enn_k=25,
        sigma_range=None,
    )
    _assert_be_config_fields(
        cfg,
        num_probes=10,
        num_candidates=10,
        warmup=20,
        fit_interval=10,
        enn_k=25,
        sigma_range=None,
    )


def test_be_config_custom_values():
    cfg = BEConfig(
        num_probes=20,
        num_candidates=5,
        warmup=50,
        fit_interval=25,
        enn_k=50,
        sigma_range=(1e-5, 1e-1),
    )
    _assert_be_config_fields(
        cfg,
        num_probes=20,
        num_candidates=5,
        warmup=50,
        fit_interval=25,
        enn_k=50,
        sigma_range=(1e-5, 1e-1),
    )


def test_be_config_frozen_dataclass():
    cfg = BEConfig(
        num_probes=10,
        num_candidates=10,
        warmup=20,
        fit_interval=10,
        enn_k=25,
        sigma_range=None,
    )
    with pytest.raises(FrozenInstanceError):
        cfg.num_probes = 15


def test_enn_config_default_creation():
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


def test_enn_config_custom_values():
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


def test_enn_config_frozen_dataclass():
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


def test_uhd_config_full_creation():
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
        sigma=0.001,
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


def test_uhd_config_frozen_dataclass():
    cfg = UHDConfig(
        env_tag="mnist",
        policy_tag=None,
        num_rounds=100,
        problem_seed=None,
        noise_seed_0=None,
        lr=0.001,
        sigma=0.001,
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
