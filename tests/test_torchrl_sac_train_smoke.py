from pathlib import Path


def test_sac_train_smoke(tmp_path):
    from problems.env_conf import get_env_conf
    from rl.env_provider import register_get_env_conf
    from rl.torchrl.sac.trainer import SACConfig, train_sac

    register_get_env_conf(get_env_conf)
    exp_dir = Path(tmp_path) / "sac_smoke"
    cfg = SACConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        seed=0,
        device="cpu",  # MPS can crash with LazyTensorStorage; use CPU for tests
        total_timesteps=96,
        learning_starts=8,
        batch_size=8,
        replay_size=2000,
        update_every=2,
        updates_per_step=1,
        eval_interval_steps=32,
        log_interval_steps=32,
        num_denoise=1,
        num_denoise_passive=1,
    )
    result = train_sac(cfg)
    metrics_path = exp_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert result.num_steps == 96
