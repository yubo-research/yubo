from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_mjx_sac_config_sections_parse() -> None:
    from rl.mjx_sac_config import MJXSACConfig, MJXSACSections
    from rl.mjx_sac_config_sections import (
        MJXSACCollectorConfig,
        MJXSACLossConfig,
        MJXSACOptimConfig,
    )

    cfg = MJXSACConfig.from_dict(
        {
            "env_tag": "mujoco_playground:CheetahRun",
            "hidden_size": 32,
            "collector": {"num_envs": 8, "num_steps": 2, "batch_size": 16},
            "optim": {"lr_actor": 1e-4},
            "loss": {"tau": 0.01},
        }
    )
    assert isinstance(cfg.sections, MJXSACSections)
    assert isinstance(cfg.collector, MJXSACCollectorConfig)
    assert isinstance(cfg.optim, MJXSACOptimConfig)
    assert isinstance(cfg.loss, MJXSACLossConfig)
    assert cfg.env_tag == "mujoco_playground:CheetahRun"
    assert cfg.hidden_size == 32
    assert cfg.collector.num_envs == 8
    assert cfg.collector.num_steps == 2
    assert cfg.collector.batch_size == 16
    assert cfg.optim.lr_actor == 1e-4
    assert cfg.loss.tau == 0.01
    assert cfg.to_dict()["env_tag"] == "mujoco_playground:CheetahRun"


def test_register_mjx_sac() -> None:
    from rl.mjx_sac import register, train_mjx_sac
    from rl.registry import get_algo

    try:
        register()
    except ValueError:
        pass
    spec = get_algo("mjx_sac")
    assert spec.config_cls.__name__ == "MJXSACConfig"
    assert spec.train_fn is train_mjx_sac


def test_MJXSACResult() -> None:
    from rl.mjx_sac import MJXSACResult

    result = MJXSACResult(best_return=1.0, last_rollout_return=0.5, num_steps=4)
    assert result.best_return == 1.0
    assert result.last_rollout_return == 0.5
    assert result.num_steps == 4


def test_mjx_sac_state_checkpoint_helpers() -> None:
    from rl import mjx_sac_state

    replay = mjx_sac_state._Replay(
        obs="obs",
        action="action",
        reward="reward",
        terminated="terminated",
        truncated="truncated",
        next_obs="next_obs",
        ptr=0,
        size=1,
    )
    state = mjx_sac_state._TrainState(
        actor={"actor": 1},
        critic1={"critic1": 1},
        critic2={"critic2": 2},
        target1={"target1": 1},
        target2={"target2": 2},
        actor_opt="actor_opt",
        critic_opt="critic_opt",
        alpha_opt="alpha_opt",
        log_alpha=0.0,
        obs_rms="obs_rms",
        reward_rms="reward_rms",
        discounted_return="discounted",
        obs="obs",
        env_state="env_state",
        running_return="running_return",
        running_length="running_length",
        replay=replay,
        key="key",
    )

    agent = mjx_sac_state._checkpoint_fn(state)
    assert isinstance(agent, mjx_sac_state._AgentState)
    assert agent.actor == {"actor": 1}
    assert agent.reward_rms == "reward_rms"


def test_mjx_sac_tanh_log_prob_keeps_squash_correction() -> None:
    from rl.mjx_ppo import _normal_log_prob
    from rl.mjx_sac import _tanh_normal_log_prob

    raw_action = np.asarray([[0.25, -0.5]], dtype=np.float32)
    mean = np.asarray([[0.1, -0.2]], dtype=np.float32)
    std = np.asarray([[0.8, 1.2]], dtype=np.float32)
    correction = np.sum(np.log(1.0 - np.tanh(raw_action) ** 2 + 1e-6), axis=-1)

    np.testing.assert_allclose(
        _tanh_normal_log_prob(np, raw_action, mean, std),
        _normal_log_prob(np, raw_action, mean, std) - correction,
    )


def test_mjx_sac_sample_action_applies_tanh_and_action_scale_log_prob() -> None:
    from rl.mjx_sac import _sample_action, _tanh_normal_log_prob

    class FakeRandom:
        @staticmethod
        def normal(_key, shape):
            assert shape == (1, 2)
            return np.asarray([[0.0, 0.0]], dtype=np.float32)

    actor = {
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
    low = np.asarray([-2.0, -4.0], dtype=np.float32)
    high = np.asarray([2.0, 0.0], dtype=np.float32)

    action, log_prob = _sample_action(SimpleNamespace(random=FakeRandom), np, actor, obs, None, low, high)

    raw = np.zeros((1, 2), dtype=np.float32)
    scale = 0.5 * (high - low)
    np.testing.assert_allclose(action, np.asarray([[0.0, -2.0]], dtype=np.float32))
    np.testing.assert_allclose(
        log_prob,
        _tanh_normal_log_prob(np, raw, raw, np.ones((2,), dtype=np.float32)) - np.sum(np.log(scale), axis=-1),
    )


def test_mjx_sac_target_masks_truncation_with_auto_reset() -> None:
    import inspect

    import rl.mjx_sac as mjx_sac

    source = inspect.getsource(mjx_sac._make_train_step)
    assert 'batch["truncated"]' in source
    assert "1.0 - done" in source


def test_train_mjx_sac_orchestrates_with_mock_runtime(monkeypatch, tmp_path) -> None:
    import rl.mjx_sac as mjx_sac
    import rl.mjx_train_loop as mjx_train_loop
    from rl.mjx_sac import _Runtime
    from rl.mjx_sac_config import MJXSACConfig

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

    cfg = MJXSACConfig.from_dict(
        {
            "exp_dir": str(tmp_path),
            "collector": {
                "total_frames": 4,
                "num_envs": 2,
                "num_steps": 1,
                "batch_size": 2,
                "updates_per_iter": 1,
            },
            "log_interval": 1,
        }
    )
    runtime = _Runtime(
        FakeJax(),
        np,
        SimpleNamespace(),
        SimpleNamespace(),
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
            "loss_actor": 0.1,
            "loss_critic": 0.2,
            "loss_alpha": 0.3,
            "alpha_value": 0.4,
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

    monkeypatch.setattr(mjx_sac, "_make_runtime", lambda _cfg: runtime)
    monkeypatch.setattr(
        mjx_train_loop,
        "_checkpoint_manager",
        lambda _ckpt_dir: FakeCheckpointManager(),
    )
    monkeypatch.setattr(
        mjx_sac,
        "_init_state",
        lambda _cfg, _runtime: (
            SimpleNamespace(
                actor={},
                critic1={},
                critic2={},
                target1={},
                target2={},
                actor_opt="actor_opt",
                critic_opt="critic_opt",
                alpha_opt="alpha_opt",
                log_alpha=0.0,
                obs_rms={},
                reward_rms={},
            ),
            ("opts",),
        ),
    )
    monkeypatch.setattr(mjx_sac, "_make_train_step", lambda _cfg, _runtime, _optimizers: fake_train_step)
    monkeypatch.setattr(
        mjx_sac,
        "make_sac_eval_step",
        lambda _cfg, _runtime: lambda *_args: float(calls["step"]),
    )

    result = mjx_sac.train_mjx_sac(cfg)

    assert result.best_return == 2.0
    assert result.last_rollout_return == 2.0
    assert result.num_steps == 4
    assert calls["step"] == 2
    assert (tmp_path / "seed_0" / "metrics.jsonl").exists()
