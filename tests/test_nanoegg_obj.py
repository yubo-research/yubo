from __future__ import annotations

import numpy as np
import pytest


def _cfg(**overrides):
    from ops.exp_uhd import _parse_cfg

    base = {
        "env_tag": "pretrain:nanoegg:synthetic",
        "policy_tag": "nanoegg:int8:1l:8d",
        "num_rounds": 1,
        "pretrain_search_dim": 8,
        "pretrain_generation_length": 8,
        "sub_dataset_size": 128,
        "problem_seed": 0,
    }
    base.update(overrides)
    return _parse_cfg(base)


def test_nanoegg_param_tree_shapes_and_score_are_finite():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from problems.nanoegg_obj import _init_params, _score_sequence

    params, es_map = _init_params(jax, jnp, seed=0, vocab_size=256, n_layer=2, n_embd=8)

    assert params["emb"].shape == (256, 8)
    assert params["blocks"]["att"]["Wf"].shape == (2, 8, 8)
    assert params["blocks"]["mlp"]["0"]["weight"].shape == (2, 32, 8)
    expected_params = 2 * 256 * 8 + 2 * (12 * 8 * 8 + 4 * 8) + 8
    assert sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)) == expected_params
    assert es_map["emb"] == 2
    assert es_map["blocks"]["ln1"]["weight"] == 0

    tokens = jnp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.uint8)
    score = _score_sequence(jax, jnp, params, tokens)
    assert np.isfinite(float(score))


def test_nanoegg_objective_decode_evaluate_and_noise_are_deterministic():
    pytest.importorskip("jax")

    from problems.nanoegg_obj import NanoEggObjectiveSpec, NanoEggUHDObjective

    obj = NanoEggUHDObjective(
        _cfg(),
        NanoEggObjectiveSpec(
            env_tag="pretrain:nanoegg:synthetic",
            dataset="synthetic",
            policy_tag="nanoegg:int8:1l:8d",
            dtype="int8",
            n_layer=1,
            n_embd=8,
        ),
    )

    mu1, se1 = obj.evaluate(obj.x0, seed=3)
    mu2, se2 = obj.evaluate(obj.x0, seed=3)
    assert (mu1, se1) == (mu2, se2)

    dense1 = obj.sample_noise(seed=7)
    dense2 = obj.sample_noise(seed=7)
    np.testing.assert_allclose(dense1, dense2)

    sparse = obj.sample_noise(seed=7, num_dim_target=2)
    assert np.count_nonzero(sparse) == 2

    policy = obj.make_policy(obj.x0)
    assert policy.x.shape == (8,)
    assert "emb" in policy.params


def test_nanoegg_decode_scale_moves_int8_params_for_small_bszo_epsilon():
    jax = pytest.importorskip("jax")

    from problems.nanoegg_obj import NanoEggObjectiveSpec, NanoEggUHDObjective

    obj = NanoEggUHDObjective(
        _cfg(
            policy_tag="nanoegg:int8:1l:16d",
            pretrain_search_dim=64,
            pretrain_delta_scale=10000.0,
        ),
        NanoEggObjectiveSpec(
            env_tag="pretrain:nanoegg:synthetic",
            dataset="synthetic",
            policy_tag="nanoegg:int8:1l:16d",
            dtype="int8",
            n_layer=1,
            n_embd=16,
        ),
    )

    noise = obj.sample_noise(seed=0)
    params0 = obj.make_policy(obj.x0).params
    params1 = obj.make_policy(obj.x0 + 1e-4 * noise).params
    diff = sum(int(np.count_nonzero(np.asarray(a) != np.asarray(b))) for a, b in zip(jax.tree_util.tree_leaves(params0), jax.tree_util.tree_leaves(params1)))

    assert diff > 0


def test_nanoegg_objective_eggroll_perturb_contract():
    pytest.importorskip("jax")

    from problems.nanoegg_obj import NanoEggObjectiveSpec, NanoEggUHDObjective

    obj = NanoEggUHDObjective(
        _cfg(),
        NanoEggObjectiveSpec(
            env_tag="pretrain:nanoegg:synthetic",
            dataset="synthetic",
            policy_tag="nanoegg:int8:1l:8d",
            dtype="int8",
            n_layer=1,
            n_embd=8,
        ),
    )

    noise = obj.sample_eggroll_noiser_noise(obj.x0, seed=11, noiser_name="eggroll")
    assert noise.shape == (8,)
    np.testing.assert_allclose(noise, obj.sample_eggroll_noiser_noise(obj.x0, seed=11, noiser_name="eggroll"))
    assert np.all(np.isfinite(noise))
    assert np.count_nonzero(noise) > 0
    dense = obj.sample_noise(seed=11)
    assert not np.allclose(noise, dense)
    with pytest.raises(ValueError, match="eggroll"):
        obj.sample_eggroll_noiser_noise(obj.x0, seed=11, noiser_name="other")


def test_nanoegg_eggroll_perturb_rank_changes_direction():
    pytest.importorskip("jax")

    from problems.nanoegg_obj import NanoEggObjectiveSpec, NanoEggUHDObjective

    obj = NanoEggUHDObjective(
        _cfg(policy_tag="nanoegg:int8:1l:16d", pretrain_search_dim=64),
        NanoEggObjectiveSpec(
            env_tag="pretrain:nanoegg:synthetic",
            dataset="synthetic",
            policy_tag="nanoegg:int8:1l:16d",
            dtype="int8",
            n_layer=1,
            n_embd=16,
        ),
    )

    r1 = obj.sample_eggroll_noiser_noise(obj.x0, seed=3, rank=1)
    r2 = obj.sample_eggroll_noiser_noise(obj.x0, seed=3, rank=2)

    assert r1.shape == (64,)
    assert r2.shape == (64,)
    assert not np.allclose(r1, r2)
