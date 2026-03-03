from types import SimpleNamespace

import numpy as np
import torch

import rl.pufferlib.wpo as puffer_wpo


def test_puffer_wpo_config_from_dict_converts_hidden_sizes():
    cfg = puffer_wpo.WPOConfig.from_dict(
        {
            "exp_dir": "_tmp/wpo_puffer_test",
            "backbone_hidden_sizes": [128, 64],
            "actor_head_hidden_sizes": [64],
            "critic_head_hidden_sizes": [64, 32],
            "vector_num_workers": 4,
        }
    )
    assert cfg.exp_dir == "_tmp/wpo_puffer_test"
    assert cfg.backbone_hidden_sizes == (128, 64)
    assert cfg.actor_head_hidden_sizes == (64,)
    assert cfg.critic_head_hidden_sizes == (64, 32)
    assert cfg.vector_num_workers == 4


def test_puffer_wpo_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn, *, backend=None):
        calls.append((name, config_cls, train_fn, backend))

    monkeypatch.setattr("rl.registry.register_algo", fake_register_algo)
    puffer_wpo.register()

    assert len(calls) == 1
    name, config_cls, train_fn, backend = calls[0]
    assert name == "wpo"
    assert backend == "pufferlib"
    assert config_cls is puffer_wpo.WPOConfig
    assert train_fn is puffer_wpo.train_wpo_puffer


def test_puffer_wpo_train_delegates_to_impl(monkeypatch):
    from rl.pufferlib.wpo import engine

    sentinel = puffer_wpo.TrainResult(
        best_return=1.0,
        last_eval_return=0.5,
        last_heldout_return=0.4,
        num_steps=12,
    )
    monkeypatch.setattr(engine, "train_wpo_puffer_impl", lambda _cfg: sentinel)

    out = engine.train_wpo_puffer(puffer_wpo.WPOConfig())
    assert out is sentinel


def test_puffer_wpo_train_impl_smoke_with_patched_loop(monkeypatch, tmp_path):
    from rl.pufferlib.wpo import engine

    cfg = puffer_wpo.WPOConfig(
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
        env_conf=SimpleNamespace(),
    )
    obs_spec = SimpleNamespace(mode="vector", raw_shape=(3,), vector_dim=3)

    class _FakeVecEnv:
        def reset(self, seed=None):
            _ = seed
            return np.zeros((1, 3), dtype=np.float32), [{}]

        def close(self):
            return None

    monkeypatch.setattr("rl.eval_noise.normalize_eval_noise_mode", lambda _mode: None)
    monkeypatch.setattr(
        engine,
        "_init_run_artifacts",
        lambda _cfg: (tmp_path / "exp", tmp_path / "exp" / "metrics.jsonl", SimpleNamespace(save_both=lambda *_args, **_kwargs: None)),
    )
    monkeypatch.setattr(engine, "_init_runtime", lambda _cfg: (env_setup, torch.device("cpu")))
    monkeypatch.setattr(engine, "make_vector_env", lambda _cfg: _FakeVecEnv())
    monkeypatch.setattr(engine, "infer_observation_spec", lambda _cfg, _obs: obs_spec)
    monkeypatch.setattr(engine, "prepare_obs_np", lambda obs_np, **_kwargs: np.asarray(obs_np, dtype=np.float32))
    monkeypatch.setattr(
        engine,
        "_build_training_components",
        lambda *_args, **_kwargs: (
            SimpleNamespace(actor_backbone=SimpleNamespace(), actor_head=SimpleNamespace()),
            SimpleNamespace(),
            SimpleNamespace(),
            engine.TrainState(start_time=0.0),
        ),
    )
    monkeypatch.setattr(engine, "_log_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("rl.logger.log_run_footer", lambda **_kwargs: None)
    monkeypatch.setattr(engine, "render_videos_if_enabled", lambda *_args, **_kwargs: None)

    def _fake_train_loop(_config, _env_setup, _modules, _optimizers, _replay, state, _obs_spec, _obs_batch, _envs, **_kwargs):
        updates = {
            "global_step": int(_config.total_timesteps),
            "best_return": 1.0,
            "last_eval_return": 0.5,
            "last_heldout_return": 0.25,
        }
        for key, value in updates.items():
            setattr(state, key, value)

    monkeypatch.setattr(engine, "_train_loop", _fake_train_loop)
    monkeypatch.setattr(engine, "use_actor_state", lambda *_args, **_kwargs: __import__("contextlib").nullcontext())

    out = puffer_wpo.train_wpo_puffer(cfg)
    assert out.num_steps == 8
    assert out.best_return == 1.0
