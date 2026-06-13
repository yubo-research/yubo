"""Extra UHD parse tests (split for kiss size limits)."""

from __future__ import annotations

import pytest

import ops.exp_uhd_parse as _exp_uhd_parse

_parse_cfg = _exp_uhd_parse._parse_cfg
_validate_required = _exp_uhd_parse._validate_required
_parse_perturb = _exp_uhd_parse._parse_perturb
_parse_perturb_spec = _exp_uhd_parse._parse_perturb_spec


def test_parse_perturb_dense():
    ndt, nmt = _parse_perturb("dense")
    assert ndt is None
    assert nmt is None
    assert _parse_perturb_spec("dense") == ("flat", None, None)


def test_parse_perturb_eggroll():
    ndt, nmt = _parse_perturb("eggroll")
    assert ndt is None
    assert nmt is None
    assert _parse_perturb_spec("eggroll") == ("eggroll", None, None)


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


def test_validate_required_raises_without_round_budget():
    cfg = {"env_tag": "mnist"}
    with pytest.raises(ValueError, match="num_rounds"):
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
    assert result.num_reps == 1
    assert result.total_timesteps is None
    assert result.problem_seed is None
    assert result.noise_seed_0 is None
    assert result.lr == 0.001
    assert result.perturb_backend == "flat"
    assert result.num_dim_target == 0.5
    assert result.num_module_target is None
    assert result.optimizer == "mezo"


def test_parse_cfg_full_config():
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


def test_parse_cfg_eggroll_perturb_config():
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


def test_total_timesteps_derives_num_rounds_for_eggroll_mezo():
    cfg = {
        "env_tag": "gymnax:CartPole-v1",
        "total_timesteps": 500_000_000,
        "steps_per_episode": 500,
        "num_envs": 8,
        "optimizer": "mezo",
    }
    result = _parse_cfg(cfg)
    assert result.total_timesteps == 500_000_000
    assert result.num_rounds == 125_000


def test_total_timesteps_derives_num_rounds_for_eggroll_bszo():
    cfg = {
        "env_tag": "gymnax:CartPole-v1",
        "total_timesteps": 1_000,
        "steps_per_episode": 10,
        "num_envs": 5,
        "optimizer": "bszo",
        "bszo_k": 4,
    }
    result = _parse_cfg(cfg)
    assert result.num_rounds == 4


def test_total_timesteps_derives_num_rounds_for_pretrain_vector_objective():
    cfg = {
        "env_tag": "pretrain:hyperscalees:gsm8k-7w3b",
        "policy_tag": "hyperscalees-rwkv-7w3b-lora-r1",
        "total_timesteps": 819_200,
        "steps_per_episode": 1,
        "num_envs": 1,
        "optimizer": "mezo_be",
    }
    result = _parse_cfg(cfg)
    assert result.num_rounds == 819_200
    assert result.policy_tag == "hyperscalees-rwkv-7w3b-lora-r1"
    assert result.pretrain_lora_only is True
    assert result.pretrain_basis_max_leaves == 32


def test_pretrain_basis_max_leaves_override_is_preserved():
    cfg = {
        "env_tag": "pretrain:hyperscalees:gsm8k-7w3b",
        "policy_tag": "hyperscalees-rwkv-7w3b-lora-r1",
        "num_rounds": 1,
        "pretrain_basis_max_leaves": 8,
    }
    result = _parse_cfg(cfg)
    assert result.pretrain_basis_max_leaves == 8


def test_pretrain_hyperscalees_overrides_parse():
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


def test_total_timesteps_only_rejects_non_vector_objective():
    cfg = {
        "env_tag": "mnist",
        "total_timesteps": 1_000,
    }
    with pytest.raises(ValueError, match="UHD vector objective"):
        _parse_cfg(cfg)


def test_total_timesteps_requires_exact_round_budget():
    cfg = {
        "env_tag": "gymnax:CartPole-v1",
        "total_timesteps": 11,
        "steps_per_episode": 5,
        "num_envs": 2,
        "optimizer": "mezo",
    }
    with pytest.raises(ValueError, match="must be divisible"):
        _parse_cfg(cfg)
