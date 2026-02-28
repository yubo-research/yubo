import pytest

from rl import builtins
from rl.pufferlib.r2d2.config import R2D2Config
from rl.pufferlib.r2d2.engine import TrainResult as R2D2TrainResult
from rl.pufferlib.r2d2.engine import train_r2d2
from rl.registry import get_algo
from rl.torchrl.tdmpc2.config import TDMPC2Config
from rl.torchrl.tdmpc2.trainer import TrainResult as TDMPC2TrainResult
from rl.torchrl.tdmpc2.trainer import train_tdmpc2


def test_tdmpc2_config_from_dict_roundtrip():
    cfg = TDMPC2Config.from_dict(
        {
            "exp_dir": "_tmp/tdmpc2",
            "env_tag": "dm:finger-turn_hard",
            "seed": "7",
            "total_timesteps": "12345",
            "horizon": "3",
            "rollout_batch_size": "128",
            "latent_dim": "64",
            "hidden_dim": "128",
            "eval_interval": "4",
            "checkpoint_interval": "20",
        }
    )
    payload = cfg.to_dict()
    assert payload["seed"] == 7
    assert payload["total_timesteps"] == 12345
    assert payload["horizon"] == 3
    assert payload["checkpoint_interval"] == 20


def test_r2d2_config_from_dict_roundtrip():
    cfg = R2D2Config.from_dict(
        {
            "exp_dir": "_tmp/r2d2",
            "env_tag": "atari:Pitfall",
            "seed": "5",
            "total_timesteps": "9999",
            "num_envs": "8",
            "unroll_length": "40",
            "burn_in": "20",
            "replay_capacity": "1000",
            "batch_size": "16",
            "target_update_interval": "500",
            "eval_interval": "2",
            "checkpoint_interval": "10",
        }
    )
    payload = cfg.to_dict()
    assert payload["seed"] == 5
    assert payload["total_timesteps"] == 9999
    assert payload["unroll_length"] == 40
    assert payload["checkpoint_interval"] == 10


def test_native_algo_registry_bindings():
    builtins.register_all()
    tdmpc2 = get_algo("tdmpc2", backend="torchrl")
    r2d2 = get_algo("r2d2", backend="pufferlib")
    assert tdmpc2.name == "tdmpc2"
    assert r2d2.name == "r2d2"


def test_tdmpc2_trainer_runs_smoke(tmp_path):
    _ = TDMPC2TrainResult(best_return=0.0, last_eval_return=0.0, num_updates=0, total_steps=0)
    cfg = TDMPC2Config(
        exp_dir=str(tmp_path / "tdmpc2"),
        env_tag="pend",
        seed=1,
        total_timesteps=48,
        warmup_steps=8,
        rollout_batch_size=8,
        updates_per_step=1,
        plan_samples=16,
        plan_elites=4,
        plan_iters=2,
        eval_interval=24,
        eval_episodes=1,
        checkpoint_interval=None,
        log_interval=24,
        device="cpu",
    )
    result = train_tdmpc2(cfg)
    assert isinstance(result, TDMPC2TrainResult)
    assert result.total_steps == 48
    assert result.num_updates > 0


def test_r2d2_trainer_runs_smoke(tmp_path):
    pytest.importorskip("ale_py")
    _ = R2D2TrainResult(best_return=0.0, last_eval_return=0.0, num_updates=0, total_steps=0)
    cfg = R2D2Config(
        exp_dir=str(tmp_path / "r2d2"),
        env_tag="atari:Pong",
        seed=1,
        total_timesteps=80,
        num_envs=2,
        unroll_length=4,
        burn_in=2,
        recurrent_hidden_dim=64,
        replay_capacity=128,
        batch_size=4,
        learning_starts=0,
        updates_per_step=1,
        target_update_interval=8,
        eval_interval=40,
        eval_episodes=1,
        checkpoint_interval=None,
        log_interval=40,
        device="cpu",
    )
    result = train_r2d2(cfg)
    assert isinstance(result, R2D2TrainResult)
    assert result.total_steps >= 80
    assert result.num_updates > 0
