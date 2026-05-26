from __future__ import annotations

from types import SimpleNamespace


def test_mjx_ppo_config_sections_parse() -> None:
    from rl.mjx_ppo_config import MJXPPOConfig, MJXPPOSections
    from rl.mjx_ppo_config_sections import MJXPPOCollectorConfig, MJXPPOLossConfig, MJXPPOOptimConfig

    cfg = MJXPPOConfig.from_dict(
        {
            "env_tag": "brax:ant",
            "hidden_size": 32,
            "collector": {"num_envs": 8, "num_steps": 2},
            "optim": {"lr": 1e-4},
            "loss": {"clip_epsilon": 0.1},
        }
    )
    assert isinstance(cfg.sections, MJXPPOSections)
    assert isinstance(cfg.collector, MJXPPOCollectorConfig)
    assert isinstance(cfg.optim, MJXPPOOptimConfig)
    assert isinstance(cfg.loss, MJXPPOLossConfig)
    assert cfg.env_tag == "brax:ant"
    assert cfg.hidden_size == 32
    assert cfg.collector.num_envs == 8
    assert cfg.collector.num_steps == 2
    assert cfg.optim.lr == 1e-4
    assert cfg.loss.clip_epsilon == 0.1
    assert cfg.to_dict()["env_tag"] == "brax:ant"


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

    result = MJXPPOResult(best_return=1.0, last_eval_return=0.5, num_iterations=4)
    assert result.best_return == 1.0
    assert result.last_eval_return == 0.5
    assert result.num_iterations == 4


def test_train_mjx_ppo_orchestrates_with_mock_runtime(monkeypatch, tmp_path) -> None:
    import rl.mjx_ppo as mjx_ppo
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
            return lambda keys: ([fn(key)[0] for key in keys], [fn(key)[1] for key in keys])

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
            "log_interval": 1,
        }
    )
    runtime = _Runtime(FakeJax(), None, fake_optax, SimpleNamespace(reset=lambda _key: ("obs", "env_state")), 3, 2)
    calls = {"step": 0}

    def fake_train_step(state):
        calls["step"] += 1
        return state, {
            "eval_return": float(calls["step"]),
            "loss": 0.0,
            "loss_objective": 0.1,
            "loss_critic": 0.2,
            "entropy": 0.3,
            "approx_kl": 0.4,
            "clipfrac": 0.5,
        }

    monkeypatch.setattr(mjx_ppo, "_make_runtime", lambda _cfg: runtime)
    monkeypatch.setattr(mjx_ppo, "_init_params", lambda *_args: {"params": 1})
    monkeypatch.setattr(mjx_ppo, "_make_train_step", lambda _cfg, _adapter, _optimizer: fake_train_step)
    monkeypatch.setattr(mjx_ppo, "_TrainState", lambda *args: args)

    result = mjx_ppo.train_mjx_ppo(cfg)

    assert result.best_return == 2.0
    assert result.last_eval_return == 2.0
    assert result.num_iterations == 2
    assert calls["step"] == 2
    assert (tmp_path / "seed_0" / "metrics.jsonl").exists()
