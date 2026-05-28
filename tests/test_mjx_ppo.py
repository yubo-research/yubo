from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_mjx_ppo_config_sections_parse() -> None:
    from rl.mjx_ppo_config import MJXPPOConfig, MJXPPOSections
    from rl.mjx_ppo_config_sections import (
        MJXPPOCollectorConfig,
        MJXPPOLossConfig,
        MJXPPOOptimConfig,
    )

    cfg = MJXPPOConfig.from_dict(
        {
            "env_tag": "mujoco_playground:CheetahRun",
            "hidden_size": 32,
            "collector": {"num_envs": 8, "num_steps": 2},
            "optim": {"lr": 1e-4, "anneal_lr": False},
            "loss": {"clip_epsilon": 0.1},
        }
    )
    assert isinstance(cfg.sections, MJXPPOSections)
    assert isinstance(cfg.collector, MJXPPOCollectorConfig)
    assert isinstance(cfg.optim, MJXPPOOptimConfig)
    assert isinstance(cfg.loss, MJXPPOLossConfig)
    assert cfg.env_tag == "mujoco_playground:CheetahRun"
    assert cfg.hidden_size == 32
    assert cfg.collector.num_envs == 8
    assert cfg.collector.num_steps == 2
    assert cfg.optim.lr == 1e-4
    assert cfg.optim.anneal_lr is False
    assert cfg.loss.clip_epsilon == 0.1
    assert cfg.to_dict()["env_tag"] == "mujoco_playground:CheetahRun"


def test_register_mjx_ppo() -> None:
    from rl.mjx_ppo import register, train_mjx_ppo
    from rl.registry import get_algo

    try:
        register()
    except ValueError:
        pass
    spec = get_algo("mjx_ppo")
    assert spec.config_cls.__name__ == "MJXPPOConfig"
    assert spec.train_fn is train_mjx_ppo


def test_MJXPPOResult() -> None:
    from rl.mjx_ppo import MJXPPOResult

    result = MJXPPOResult(best_return=1.0, last_rollout_return=0.5, num_iterations=4)
    assert result.best_return == 1.0
    assert result.last_rollout_return == 0.5
    assert result.num_iterations == 4


def test_mjx_ppo_loop_state_and_record_helpers() -> None:
    from rl import mjx_ppo_loop

    state = mjx_ppo_loop._TrainState(
        iteration=3,
        params={"p": 1},
        opt_state="opt",
        obs_rms="obs_rms",
        reward_rms="reward_rms",
        discounted_return="discounted",
        obs="obs",
        env_state="env_state",
        running_return="running_return",
        running_length="running_length",
        key="key",
    )

    agent = mjx_ppo_loop._checkpoint_fn(state)
    assert isinstance(agent, mjx_ppo_loop._AgentState)
    assert agent.params == {"p": 1}
    assert mjx_ppo_loop._eval_args(state) == ({"p": 1}, "obs_rms")

    result = mjx_ppo_loop._result(2.0, 1.5, 7, 64)
    assert isinstance(result, mjx_ppo_loop.MJXPPOResult)
    assert result.best_return == 2.0
    assert result.last_rollout_return == 1.5
    assert result.num_iterations == 7

    metrics = {
        "rollout_return": 1.0,
        "ep_ret": 1.0,
        "ep_len": 2.0,
        "rollout_reward": 0.25,
        "loss": 0.0,
        "loss_objective": 0.1,
        "loss_critic": 0.2,
        "entropy": 0.3,
        "approx_kl": 0.4,
        "clipfrac": 0.5,
        "done_fraction": 0.6,
    }
    record = mjx_ppo_loop._ppo_iter_record(
        iteration=2,
        frames_per_iter=32,
        elapsed=4.0,
        iter_dt=0.5,
        metrics=metrics,
        ret_best=3.0,
    )
    assert record["step"] == 64
    assert record["fps"] == 64.0
    assert record["ret_best"] == 3.0


def test_mjx_ppo_normal_log_prob_is_unsquashed() -> None:
    from rl.mjx_ppo import _normal_log_prob

    action = np.asarray([[0.25, -0.5]], dtype=np.float32)
    mean = np.asarray([[0.1, -0.2]], dtype=np.float32)
    std = np.asarray([[0.8, 1.2]], dtype=np.float32)
    z = (action - mean) / std
    expected = np.sum(
        -0.5 * z * z - np.log(std) - 0.5 * np.log(2.0 * np.pi),
        axis=-1,
    )

    np.testing.assert_allclose(_normal_log_prob(np, action, mean, std), expected)
    tanh_correction = np.sum(np.log(1.0 - np.tanh(action) ** 2 + 1e-6), axis=-1)
    assert _normal_log_prob(np, action, mean, std)[0] != pytest.approx(expected[0] - tanh_correction[0])


def test_mjx_ppo_loss_and_learning_rate_match_clipped_surrogate() -> None:
    from rl.mjx_ppo import _init_params, _ppo_learning_rate, _ppo_loss

    config = SimpleNamespace(
        optim=SimpleNamespace(lr=0.01, anneal_lr=True),
        loss=SimpleNamespace(
            clip_epsilon=0.2,
            clip_value_loss=True,
            critic_coeff=0.5,
            entropy_coeff=0.01,
        ),
    )
    assert _ppo_learning_rate(np, config, iteration=1, num_iterations=10) == pytest.approx(0.01)
    assert _ppo_learning_rate(np, config, iteration=10, num_iterations=10) == pytest.approx(0.001)

    params = _init_params(
        SimpleNamespace(
            random=SimpleNamespace(
                split=lambda _key, count=2: tuple(range(count)),
                normal=lambda _key, shape: np.zeros(shape, dtype=np.float32),
            )
        ),
        np,
        key=0,
        obs_dim=3,
        act_dim=2,
        hidden=4,
    )
    rms = SimpleNamespace(
        mean=np.zeros((3,), dtype=np.float32),
        var=np.ones((3,), dtype=np.float32),
    )
    batch = {
        "obs": np.zeros((2, 3), dtype=np.float32),
        "action": np.zeros((2, 2), dtype=np.float32),
        "log_prob": np.zeros((2,), dtype=np.float32),
        "advantage": np.ones((2,), dtype=np.float32),
        "value": np.zeros((2,), dtype=np.float32),
        "target": np.ones((2,), dtype=np.float32),
    }

    loss, (loss_pi, loss_v, entropy, ratio, logratio) = _ppo_loss(np, config, params, rms, batch)

    assert np.isfinite(loss)
    assert loss_pi < 0.0
    assert loss_v == pytest.approx(1.0)
    assert entropy > 0.0
    assert ratio.shape == (2,)
    assert logratio.shape == (2,)


def test_mjx_ppo_sample_action_stores_raw_and_clips_env_action() -> None:
    from rl.mjx_ppo import _normal_log_prob, _sample_action

    def fake_normal(_key, shape):
        assert shape == (1, 2)
        return np.asarray([[2.0, -3.0]], dtype=np.float32)

    params = {
        "policy_1": {
            "w": np.zeros((3, 4), dtype=np.float32),
            "b": np.zeros((4,), dtype=np.float32),
        },
        "policy_2": {
            "w": np.zeros((4, 4), dtype=np.float32),
            "b": np.zeros((4,), dtype=np.float32),
        },
        "policy_mean": {
            "w": np.zeros((4, 2), dtype=np.float32),
            "b": np.zeros((2,), dtype=np.float32),
        },
        "log_std": np.zeros((2,), dtype=np.float32),
    }
    obs = np.zeros((1, 3), dtype=np.float32)
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)

    action, env_action, log_prob = _sample_action(
        SimpleNamespace(random=SimpleNamespace(normal=fake_normal)),
        np,
        params,
        obs,
        None,
        low,
        high,
    )

    expected_action = np.asarray([[2.0, -3.0]], dtype=np.float32)
    np.testing.assert_allclose(action, expected_action)
    np.testing.assert_allclose(env_action, np.asarray([[1.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(
        log_prob,
        _normal_log_prob(np, expected_action, np.zeros((1, 2)), np.ones((2,))),
    )


def test_mjx_ppo_train_step_masks_truncation_with_auto_reset() -> None:
    import inspect

    import rl.mjx_ppo as mjx_ppo

    source = inspect.getsource(mjx_ppo._make_train_step)
    assert 'transition["truncated"]' in source
    assert "mask = 1.0 - done" in source


def test_train_mjx_ppo_orchestrates_with_mock_runtime(monkeypatch, tmp_path) -> None:
    import rl.mjx_ppo as mjx_ppo
    import rl.mjx_train_loop as mjx_train_loop
    from rl.mjx_ppo import _Runtime
    from rl.mjx_ppo_config import MJXPPOConfig

    class FakeJax:
        random = SimpleNamespace(
            key=lambda seed: ("key", seed),
            split=lambda key, count=2: tuple(("key", idx) for idx in range(count)),
        )

        @staticmethod
        def device_get(value):
            return value

        @staticmethod
        def default_backend():
            return "fake"

        @staticmethod
        def vmap(fn):
            return lambda keys: (
                [fn(key)[0] for key in keys],
                [fn(key)[1] for key in keys],
            )

    class FakeOptimizer:
        @staticmethod
        def init(_params):
            return "opt_state"

    fake_optax = SimpleNamespace(
        chain=lambda *_args: FakeOptimizer(),
        clip_by_global_norm=lambda _value: "clip",
        adam=lambda _value: "adam",
    )

    cfg = MJXPPOConfig.from_dict(
        {
            "exp_dir": str(tmp_path),
            "collector": {"total_frames": 4, "num_envs": 2, "num_steps": 1},
            "optim": {"minibatch_size": 2, "num_epochs": 1},
            "eval": {"interval": 0},
            "log_interval": 1,
        }
    )
    runtime = _Runtime(
        FakeJax(),
        np,
        fake_optax,
        SimpleNamespace(reset=lambda _key: ("obs", "env_state")),
        3,
        2,
        np.asarray([-1.0, -1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )
    calls = {"step": 0}

    def fake_train_step(state):
        calls["step"] += 1
        return state, {
            "rollout_return": float(calls["step"]),
            "rollout_reward": 0.25,
            "ep_ret": float(calls["step"]),
            "ep_len": 1.0,
            "done_fraction": 0.5,
            "loss": 0.0,
            "loss_objective": 0.1,
            "loss_critic": 0.2,
            "entropy": 0.3,
            "approx_kl": 0.4,
            "clipfrac": 0.5,
        }

    class FakeCheckpointManager:
        @staticmethod
        def save(_step, _state):
            return None

        @staticmethod
        def wait_until_finished():
            return None

        @staticmethod
        def latest_step():
            return 0

        @staticmethod
        def restore(_step, *, items):
            return items

    monkeypatch.setattr(mjx_ppo, "_make_runtime", lambda _cfg: runtime)
    monkeypatch.setattr(
        mjx_train_loop.ocp,
        "CheckpointManager",
        lambda *_args, **_kwargs: FakeCheckpointManager(),
    )
    monkeypatch.setattr(mjx_train_loop.ocp, "AsyncCheckpointer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mjx_train_loop.ocp,
        "PyTreeCheckpointHandler",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(mjx_train_loop.ocp, "CheckpointManagerOptions", lambda **_kwargs: object())
    monkeypatch.setattr(
        mjx_ppo,
        "_init_train_state",
        lambda _cfg, _runtime: (
            SimpleNamespace(
                iteration=0,
                params={},
                opt_state="opt_state",
                obs_rms={},
                reward_rms={},
            ),
            FakeOptimizer(),
        ),
    )
    monkeypatch.setattr(mjx_ppo, "_make_train_step", lambda _cfg, _adapter, _optimizer: fake_train_step)
    monkeypatch.setattr(
        mjx_ppo,
        "_make_eval_step",
        lambda _cfg, _runtime: lambda *_args: pytest.fail("eval should be disabled"),
    )

    result = mjx_ppo.train_mjx_ppo(cfg)

    assert result.best_return == 2.0
    assert result.last_rollout_return == 2.0
    assert result.num_iterations == 2
    assert calls["step"] == 2
    assert (tmp_path / "seed_0" / "metrics.jsonl").exists()
