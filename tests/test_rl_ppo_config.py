"""Tests for PPO config types. Uses shallow imports to keep dependency depth low."""

from rl.backends.torchrl.ppo.config import PPOConfig, TrainResult


def test_train_result_dataclass_fields():
    result = TrainResult(
        best_return=1.5,
        last_eval_return=1.0,
        last_heldout_return=None,
        num_iterations=3,
    )
    assert result.best_return == 1.5
    assert result.last_eval_return == 1.0
    assert result.last_heldout_return is None
    assert result.num_iterations == 3


def test_ppo_config_runtime_num_envs_helper():
    cfg = PPOConfig(num_envs=7)
    assert cfg.runtime_num_envs() == 7
    assert PPOConfig.runtime_num_envs(cfg) == 7
