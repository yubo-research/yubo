from pathlib import Path

from rl.algos.torchrl_ppo import PPOConfig, train_ppo


def test_ppo_train_smoke(tmp_path):
    exp_dir = Path(tmp_path) / "ppo_smoke"
    cfg = PPOConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        seed=0,
        total_timesteps=64,
        num_steps=16,
        num_envs=1,
        update_epochs=1,
        num_minibatches=1,
        eval_interval=1,
        num_denoise_eval=1,
        num_denoise_passive_eval=1,
        backbone_hidden_sizes=(8,),
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
    )
    result = train_ppo(cfg)
    metrics_path = exp_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert result.num_iterations == 4
