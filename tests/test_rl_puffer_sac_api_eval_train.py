def test_train_sac_puffer_impl_smoke_with_patched_loop(monkeypatch, tmp_path):
    import importlib
    import time
    from types import SimpleNamespace

    import numpy as np
    import torch

    puffer_sac = importlib.import_module("rl.pufferlib.sac")
    train_impl = importlib.import_module("rl.pufferlib.sac.sac_puffer_train_run_impl")
    eval_utils = importlib.import_module("rl.pufferlib.sac.eval_utils")
    TrainState = eval_utils.TrainState

    class _FakeVecEnv:
        def __init__(self, num_envs: int, obs_dim: int):
            self.num_envs = int(num_envs)
            self.obs_dim = int(obs_dim)

        def reset(self, seed=None):
            _ = seed
            obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
            return obs, [{} for _ in range(self.num_envs)]

        def close(self):
            return None

    cfg = puffer_sac.SACConfig(
        exp_dir=str(tmp_path / "exp"),
        total_timesteps=8,
        num_envs=1,
        replay_size=32,
        checkpoint_interval_steps=None,
        video_enable=False,
    )
    env_setup = SimpleNamespace(
        problem_seed=7,
        act_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    obs_spec = SimpleNamespace(mode="vector", raw_shape=(3,), vector_dim=3)

    monkeypatch.setattr("rl.eval_noise.normalize_eval_noise_mode", lambda _mode: None)
    monkeypatch.setattr(
        train_impl,
        "_init_run_artifacts",
        lambda _cfg: (
            tmp_path / "exp",
            tmp_path / "exp" / "metrics.jsonl",
            SimpleNamespace(save_both=lambda *_args, **_kwargs: None),
        ),
    )
    monkeypatch.setattr(train_impl, "_init_runtime", lambda _cfg: (env_setup, torch.device("cpu")))
    monkeypatch.setattr(
        "rl.pufferlib.sac.env_utils.make_vector_env",
        lambda _cfg: _FakeVecEnv(num_envs=1, obs_dim=3),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.env_utils.infer_observation_spec",
        lambda _cfg, _obs: obs_spec,
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.env_utils.prepare_obs_np",
        lambda obs_np, **_kwargs: np.asarray(obs_np, dtype=np.float32),
    )
    monkeypatch.setattr(
        train_impl,
        "_build_training_components",
        lambda *_args, **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            TrainState(start_time=float(time.time()) - 1.0),
        ),
    )
    monkeypatch.setattr(train_impl, "_log_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("rl.logger.log_run_footer", lambda **_kwargs: None)
    monkeypatch.setattr(
        "rl.pufferlib.sac.eval_utils.render_videos_if_enabled",
        lambda *_args, **_kwargs: None,
    )

    def _fake_train_loop(
        _config,
        _env_setup,
        _modules,
        _optimizers,
        _replay,
        state,
        _obs_spec,
        _obs_batch,
        _envs,
        **_kwargs,
    ):
        state.global_step = int(_config.total_timesteps)
        state.best_return = 1.0
        state.last_eval_return = 0.5
        state.last_heldout_return = 0.25

    monkeypatch.setattr("rl.pufferlib.sac.sac_loop_impl._train_loop", _fake_train_loop)

    out = puffer_sac.train_sac_puffer(cfg)
    assert out.num_steps == 8
    assert out.best_return == 1.0
