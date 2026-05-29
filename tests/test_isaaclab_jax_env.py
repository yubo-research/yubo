from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from isaaclab_score_fakes import FakeIsaacEnv, make_fake_runtime, make_fake_vector_runtime


def test_eggroll_jax_sim_enabled_parses_opt_name():
    from problems.eggroll_jax_flags import eggroll_jax_sim_enabled

    assert eggroll_jax_sim_enabled("eggroll/jax_sim=true") is True
    assert eggroll_jax_sim_enabled("eggroll") is False
    assert eggroll_jax_sim_enabled("turbo-enn") is False


def test_isaaclab_jax_adapter_reset_and_step(monkeypatch):
    from problems.isaaclab_jax_env import IsaacLabJaxAdapter

    monkeypatch.setattr(
        "problems.isaaclab_jax_env.make_isaaclab_env",
        lambda *_args, **_kwargs: FakeIsaacEnv(),
    )
    adapter = IsaacLabJaxAdapter(
        "isaaclab:Fake-v0",
        jax=jax,
        jnp=jnp,
        problem_seed=0,
        max_steps=3,
    )
    try:
        key = jax.random.key(0)
        obs, state = adapter.reset(key)
        assert obs.shape == (2,)
        assert int(state.episode_step) == 0
        action = jnp.asarray([0.25], dtype=jnp.float32)
        result = adapter.step(key, state, action)
        assert result.obs.shape == (2,)
        assert float(result.reward) > 0.0
    finally:
        adapter.close()


def test_isaaclab_jax_adapter_reset_and_step_under_scan(monkeypatch):
    from problems.isaaclab_jax_env import IsaacLabJaxAdapter

    monkeypatch.setattr(
        "problems.isaaclab_jax_env.make_isaaclab_env",
        lambda *_args, **_kwargs: FakeIsaacEnv(),
    )
    adapter = IsaacLabJaxAdapter(
        "isaaclab:Fake-v0",
        jax=jax,
        jnp=jnp,
        problem_seed=0,
        max_steps=3,
    )
    try:

        def _scan_body(carry, _):
            key, state = carry
            action = jnp.zeros((1,), dtype=jnp.float32)
            result = adapter.step(key, state, action)
            next_carry = (key, result.state)
            return next_carry, result.reward

        key = jax.random.key(0)
        obs, state = adapter.reset(key)
        (_, _), rewards = jax.lax.scan(_scan_body, (key, state), None, length=2)
        assert rewards.shape == (2,)
        assert jnp.all(rewards >= 0.0)
    finally:
        adapter.close()


def test_isaaclab_jax_adapter_reset_and_step_under_vmap(monkeypatch):
    from problems.isaaclab_jax_env import IsaacLabJaxAdapter

    monkeypatch.setattr(
        "problems.isaaclab_jax_env.make_isaaclab_env",
        lambda *_args, **_kwargs: FakeIsaacEnv(),
    )
    adapter = IsaacLabJaxAdapter(
        "isaaclab:Fake-v0",
        jax=jax,
        jnp=jnp,
        problem_seed=0,
        max_steps=3,
    )
    try:

        def _one_rollout(key):
            obs, state = adapter.reset(key)
            action = jnp.asarray([0.25], dtype=jnp.float32)
            result = adapter.step(key, state, action)
            return obs, result.reward

        keys = jax.random.split(jax.random.key(0), 2)
        obs, rewards = jax.vmap(_one_rollout)(keys)
        assert obs.shape == (2, 2)
        assert jnp.all(rewards > 0.0)
    finally:
        adapter.close()


def test_make_jax_env_adapter_routes_isaaclab(monkeypatch):
    from problems.jax_env_factory import make_jax_env_adapter

    monkeypatch.setattr(
        "problems.isaaclab_jax_env.make_isaaclab_env",
        lambda *_args, **_kwargs: FakeIsaacEnv(),
    )
    runtime = make_fake_runtime()
    adapter = make_jax_env_adapter("isaaclab:Fake-v0", jax=jax, jnp=jnp, env_runtime=runtime)
    try:
        assert adapter.__class__.__name__ == "IsaacLabJaxAdapter"
    finally:
        adapter.close()


def test_eggroll_external_scoring_skipped_when_jax_sim():
    from optimizer.eggroll_designer import _should_use_external_scoring
    from optimizer.eggroll_designer_config import _EggRollDesignerConfig

    env_conf = SimpleNamespace(env_name="isaaclab:Fake-v0")
    cfg = _EggRollDesignerConfig(jax_sim=True)
    assert _should_use_external_scoring(env_conf, cfg) is False
    cfg_ext = _EggRollDesignerConfig(jax_sim=False)
    assert _should_use_external_scoring(env_conf, cfg_ext) is True


def test_eggroll_policy_factory_uses_jax_for_isaac_jax_sim():
    from policies.eggroll_policy import (
        EggRollActorCriticMLPPolicy,
        EggRollActorCriticMLPPolicyFactory,
        EggRollActorCriticMLPSpec,
    )

    runtime = make_fake_runtime()
    runtime.eggroll_jax_sim = True
    factory = EggRollActorCriticMLPPolicyFactory(EggRollActorCriticMLPSpec(hidden_dim=4, layers=2))
    policy = factory(runtime)
    assert isinstance(policy, EggRollActorCriticMLPPolicy)


def test_validate_eggroll_jax_allows_isaaclab():
    from optimizer.eggroll_env_validation import validate_eggroll_jax_objective_env

    validate_eggroll_jax_objective_env("isaaclab:Fake-v0", ValueError)


def test_isaaclab_jax_vector_adapter_reset_and_step_batch(monkeypatch):
    from problems.isaaclab_jax_vector_env import IsaacLabJaxVectorAdapter

    runtime = make_fake_vector_runtime()
    runtime.eggroll_population = 0
    runtime.eggroll_eval_envs = 4
    monkeypatch.setattr(
        "problems.isaaclab_jax_vector_env.make_isaaclab_env",
        lambda *_args, **kwargs: runtime.make(**kwargs),
    )
    adapter = IsaacLabJaxVectorAdapter(
        "isaaclab:Fake-v0",
        jax=jax,
        jnp=jnp,
        num_envs=4,
        problem_seed=0,
        max_steps=3,
    )
    try:
        assert adapter.vector_num_envs == 4
        key = jax.random.key(0)
        obs, state = adapter.reset(key)
        assert obs.shape == (4, 2)
        assert state.episode_step.shape == (4,)
        actions = jnp.zeros((4, 1), dtype=jnp.float32)
        next_obs, next_state, reward, term, trunc = adapter.step_batched(state, actions)
        assert next_obs.shape == (4, 2)
        assert reward.shape == (4,)
        assert term.shape == (4,)
        assert trunc.shape == (4,)
        assert next_state.episode_step.shape == (4,)
        assert runtime.vector_slots == [4]
    finally:
        adapter.close()


def test_make_isaaclab_jax_adapter_uses_vector_when_population_set(monkeypatch):
    from problems.isaaclab_jax_env import make_isaaclab_jax_adapter

    runtime = make_fake_vector_runtime()
    runtime.eggroll_population = 8
    runtime.eggroll_eval_envs = 2

    def fake_make(*_args, **_kwargs):
        return runtime.make(**_kwargs)

    monkeypatch.setattr("problems.isaaclab_jax_env.make_isaaclab_env", fake_make)
    monkeypatch.setattr("problems.isaaclab_jax_vector_env.make_isaaclab_env", fake_make)
    adapter = make_isaaclab_jax_adapter(
        "isaaclab:Fake-v0",
        jax=jax,
        jnp=jnp,
        env_runtime=runtime,
    )
    try:
        assert adapter.__class__.__name__ == "IsaacLabJaxVectorAdapter"
        assert adapter.vector_num_envs == 8
    finally:
        adapter.close()
