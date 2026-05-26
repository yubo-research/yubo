from __future__ import annotations

from types import SimpleNamespace


def test_mjx_sac_config_sections_parse() -> None:
    from rl.mjx_sac_config import MJXSACConfig, MJXSACSections
    from rl.mjx_sac_config_sections import MJXSACCollectorConfig, MJXSACLossConfig, MJXSACOptimConfig

    cfg = MJXSACConfig.from_dict(
        {
            "env_tag": "brax:ant",
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
    assert cfg.env_tag == "brax:ant"
    assert cfg.hidden_size == 32
    assert cfg.collector.num_envs == 8
    assert cfg.collector.num_steps == 2
    assert cfg.collector.batch_size == 16
    assert cfg.optim.lr_actor == 1e-4
    assert cfg.loss.tau == 0.01
    assert cfg.to_dict()["env_tag"] == "brax:ant"


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

    result = MJXSACResult(best_return=1.0, last_eval_return=0.5, num_steps=4)
    assert result.best_return == 1.0
    assert result.last_eval_return == 0.5
    assert result.num_steps == 4


def test_train_mjx_sac_orchestrates_with_mock_runtime(monkeypatch, tmp_path) -> None:
    import rl.mjx_sac as mjx_sac
    from rl.mjx_ppo import _Runtime
    from rl.mjx_sac_config import MJXSACConfig

    class FakeJax:
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
    runtime = _Runtime(FakeJax(), None, None, SimpleNamespace(), 3, 2)
    calls = {"step": 0}

    def fake_train_step(state):
        calls["step"] += 1
        return state, {
            "eval_return": float(calls["step"]),
            "loss_actor": 0.1,
            "loss_critic": 0.2,
            "loss_alpha": 0.3,
        }

    monkeypatch.setattr(mjx_sac, "_make_runtime", lambda _cfg: runtime)
    monkeypatch.setattr(mjx_sac, "_init_state", lambda _cfg, _runtime: ("state", ("opts",)))
    monkeypatch.setattr(mjx_sac, "_make_train_step", lambda _cfg, _runtime, _optimizers: fake_train_step)

    result = mjx_sac.train_mjx_sac(cfg)

    assert result.best_return == 2.0
    assert result.last_eval_return == 2.0
    assert result.num_steps == 4
    assert calls["step"] == 2
    assert (tmp_path / "seed_0" / "metrics.jsonl").exists()
