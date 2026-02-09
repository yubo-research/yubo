from pathlib import Path

from rl.algos.torchrl_sac import SACConfig, train_sac


def test_sac_train_smoke(tmp_path):
    exp_dir = Path(tmp_path) / "sac_smoke"
    cfg = SACConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        seed=0,
        total_timesteps=96,
        learning_starts=8,
        batch_size=8,
        replay_size=2000,
        update_every=2,
        updates_per_step=1,
        eval_interval_steps=32,
        log_interval_steps=32,
        num_denoise_eval=1,
        num_denoise_passive_eval=1,
    )
    result = train_sac(cfg)
    metrics_path = exp_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert result.num_steps == 96
