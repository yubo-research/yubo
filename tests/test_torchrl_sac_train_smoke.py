from pathlib import Path


def test_sac_train_smoke(tmp_path):
    """BO-style: import inside test to defer heavy deps (matches test_turbo_ackley, test_mcmc_bo)."""
    from rl.torchrl.sac import train_sac
    from rl.torchrl.sac.config import (
        SACCollectorConfig,
        SACConfig,
        SACEvalConfig,
        SACOptimConfig,
        SACReplayBufferConfig,
    )

    exp_dir = Path(tmp_path) / "sac_smoke"
    cfg = SACConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        policy_tag="mlp-16-8",
        seed=0,
        device="cpu",  # MPS can crash with LazyTensorStorage; use CPU for tests
        log_interval_steps=32,
        collector=SACCollectorConfig(total_frames=96, init_random_frames=8),
        replay_buffer=SACReplayBufferConfig(batch_size=8, size=2000),
        optim=SACOptimConfig(update_every=2, optim_steps_per_batch=1),
        eval=SACEvalConfig(interval_steps=32, num_denoise=1, num_denoise_passive=1),
    )
    result = train_sac(cfg)
    metrics_path = exp_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert result.num_steps == 96
