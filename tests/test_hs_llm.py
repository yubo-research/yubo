from __future__ import annotations

import pickle

import numpy as np
import pytest


def test_load_hyperscalees_model_retries_corrupt_cache():
    from problems.pre_obj import HyperscaleESPretrainSpec, _load_hyperscalees_model

    calls = []

    def fake_get_model(model_choice, *, rwkv_type, verbose, dtype, reload_cache=False):
        calls.append(
            {
                "model_choice": model_choice,
                "rwkv_type": rwkv_type,
                "verbose": verbose,
                "dtype": dtype,
                "reload_cache": reload_cache,
            }
        )
        if not reload_cache:
            raise pickle.UnpicklingError("pickle data was truncated")
        return "rwkv", "params", "tokenizer"

    spec = HyperscaleESPretrainSpec(
        env_tag="pretrain:hyperscalees:gsm8k-7w3b",
        task="gsm8k",
        model_choice="7w3B",
        thinking_length=256,
        answer_length=256,
    )

    assert _load_hyperscalees_model(fake_get_model, spec) == ("rwkv", "params", "tokenizer")
    assert [call["reload_cache"] for call in calls] == [False, True]


def test_subspace_codec_zero_decode_returns_base_params_and_respects_lora_map():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from problems.pre_obj import _SubspaceParamCodec

    params = {"lora": jnp.ones((8,), dtype=jnp.float32), "full": jnp.ones((8,), dtype=jnp.float32)}
    es_map = {"lora": 1, "full": 0}
    codec = _SubspaceParamCodec(
        jax,
        jnp,
        params,
        es_map=es_map,
        dim=4,
        delta_scale=1.0,
        seed=0,
        lora_only=True,
        basis_max_leaves=None,
    )

    assert codec.num_candidate_leaves == 1
    assert codec.decode(codec.x0) is params
    decoded = codec.decode(np.ones((4,), dtype=np.float64))
    np.testing.assert_allclose(np.asarray(decoded["full"]), np.ones((8,), dtype=np.float32))


def test_resolve_hyperscalees_pretrain_spec_supports_all_upstream_task_families():
    from problems.pre_obj import resolve_hyperscalees_pretrain_spec, supported_hyperscalees_llm_bandit_tasks

    tasks = supported_hyperscalees_llm_bandit_tasks()
    assert "fastzero" in tasks
    assert "uniquetok" in tasks
    assert "reptok" in tasks
    assert "digits" in tasks
    assert "gsm8ksft" in tasks
    assert "aime24" in tasks
    assert "aime25" in tasks
    assert "basic_arithmetic" in tasks
    assert "zebra_puzzles" in tasks

    spec = resolve_hyperscalees_pretrain_spec("pretrain:hyperscalees:basic_arithmetic-7w1p5b")
    assert spec.task == "basic_arithmetic"
    assert spec.model_choice == "7w1.5B"
    assert spec.thinking_length == 100
    assert spec.answer_length == 100

    spec = resolve_hyperscalees_pretrain_spec("pretrain:hyperscalees:aime25-7g7b")
    assert spec.task == "aime25"
    assert spec.model_choice == "7g7B"


def test_resolve_hyperscalees_pretrain_spec_keeps_existing_paper_tags_tuned():
    from problems.pre_obj import resolve_hyperscalees_pretrain_spec

    gsm8k = resolve_hyperscalees_pretrain_spec("pretrain:hyperscalees:gsm8k-7w3b")
    countdownn = resolve_hyperscalees_pretrain_spec("pretrain:hyperscalees:countdownn-7w1p5b")

    assert gsm8k.task == "gsm8k"
    assert gsm8k.model_choice == "7w3B"
    assert gsm8k.thinking_length == 256
    assert gsm8k.answer_length == 256
    assert countdownn.task == "countdownn"
    assert countdownn.model_choice == "7w1.5B"
    assert countdownn.thinking_length == 100
    assert countdownn.answer_length == 100


def test_resolve_hyperscalees_pretrain_spec_rejects_unknown_dynamic_task_and_model():
    import pytest

    from problems.pre_obj import resolve_hyperscalees_pretrain_spec

    with pytest.raises(ValueError, match="Unsupported HyperscaleES LLM bandit task"):
        resolve_hyperscalees_pretrain_spec("pretrain:hyperscalees:not_a_task-7w1p5b")
    with pytest.raises(ValueError, match="Unsupported HyperscaleES model tag"):
        resolve_hyperscalees_pretrain_spec("pretrain:hyperscalees:fastzero-notamodel")


def test_nanoegg_pretrain_is_real_uhd_objective_not_eggroll_surrogate():
    from problems.eggroll_env_adapters import supports_eggroll_env_adapter
    from problems.pre_obj import resolve_nanoegg_pretrain_spec
    from problems.uhd_obj import supports_uhd_vector_objective

    env_tag = "pretrain:nanoegg:minipile-int8-6l256d"
    spec = resolve_nanoegg_pretrain_spec(env_tag)

    assert supports_eggroll_env_adapter(env_tag) is False
    assert supports_uhd_vector_objective(env_tag) is True
    assert spec.dataset == "minipile"
    assert spec.dtype == "int8"
    assert spec.n_layer == 6
    assert spec.n_embd == 256


def test_nanoegg_pretrain_loads_installed_uhd_objective(monkeypatch):
    import sys
    import types

    from ops.exp_uhd import _parse_cfg
    from problems.uhd_obj import build_uhd_vector_objective

    class Obj:
        dim = 3
        x0 = np.zeros(3, dtype=np.float64)
        steps_per_episode = 1
        eval_episodes = 1

        def evaluate(self, x, *, seed):
            return float(np.sum(x) + seed), 0.25

        def evaluate_many(self, x_batch, *, seed):
            mus = np.asarray([np.sum(x) + seed + i for i, x in enumerate(x_batch)], dtype=np.float64)
            return mus, np.full(len(mus), 0.25, dtype=np.float64)

    package = types.ModuleType("nanoegg")
    package.__path__ = []
    module = types.ModuleType("nanoegg.uhd")

    def build_uhd_objective(*, cfg, spec):
        assert spec.dataset == "minipile"
        return Obj()

    module.build_uhd_objective = build_uhd_objective
    monkeypatch.setitem(sys.modules, "nanoegg", package)
    monkeypatch.setitem(sys.modules, "nanoegg.uhd", module)

    cfg = _parse_cfg(
        {
            "env_tag": "pretrain:nanoegg:minipile-int8-6l256d",
            "policy_tag": "nanoegg-int8-6l-256d",
            "num_rounds": 1,
        }
    )
    built = build_uhd_vector_objective(cfg)

    assert built.source == "nanoegg-pretrain"
    assert built.objective.dim == 3
    assert built.objective.evaluate(np.asarray([1.0, 2.0, 3.0]), seed=4) == (10.0, 0.25)


def test_nanoegg_pretrain_requires_real_runtime(monkeypatch):
    import sys

    from ops.exp_uhd import _parse_cfg
    from problems.uhd_obj import build_uhd_vector_objective

    monkeypatch.delitem(sys.modules, "nanoegg", raising=False)
    monkeypatch.delitem(sys.modules, "nanoegg.uhd", raising=False)
    cfg = _parse_cfg(
        {
            "env_tag": "pretrain:nanoegg:minipile-int8-6l256d",
            "policy_tag": "nanoegg-int8-6l-256d",
            "num_rounds": 1,
        }
    )

    with pytest.raises(ImportError, match="Real NanoEgg pretraining UHD requires"):
        build_uhd_vector_objective(cfg)
