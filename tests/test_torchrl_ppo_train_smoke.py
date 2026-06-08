from pathlib import Path


def test_ppo_train_smoke(tmp_path):
    """BO-style: import inside test to defer heavy deps (matches test_ackley_3d_runs_with_optimizer)."""
    from rl.torchrl.ppo.config import PPOCollectorConfig, PPOEvalConfig, PPOOptimConfig
    from rl.torchrl.ppo.core import PPOConfig, train_ppo

    exp_dir = Path(tmp_path) / "ppo_smoke"
    cfg = PPOConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        policy_tag="mlp-16-8",
        seed=0,
        collector=PPOCollectorConfig(total_frames=64, frames_per_batch=16, num_envs=1),
        optim=PPOOptimConfig(num_epochs=1, minibatch_size=16),
        eval=PPOEvalConfig(interval=1, num_denoise=1, num_denoise_passive=1),
    )
    result = train_ppo(cfg)
    metrics_path = exp_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert result.num_iterations == 4
