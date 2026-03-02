import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import rl.pufferlib.sac as puffer_sac


def test_puffer_sac_config_from_dict_converts_hidden_sizes():
    cfg = puffer_sac.SACConfig.from_dict(
        {
            "exp_dir": "_tmp/sac_puffer_test",
            "backbone_hidden_sizes": [128, 64],
            "actor_head_hidden_sizes": [64],
            "critic_head_hidden_sizes": [64, 32],
            "vector_num_workers": 4,
        }
    )
    assert cfg.exp_dir == "_tmp/sac_puffer_test"
    assert cfg.backbone_hidden_sizes == (128, 64)
    assert cfg.actor_head_hidden_sizes == (64,)
    assert cfg.critic_head_hidden_sizes == (64, 32)
    assert cfg.vector_num_workers == 4


def test_puffer_sac_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn, *, backend=None):
        calls.append((name, config_cls, train_fn, backend))

    monkeypatch.setattr("rl.registry.register_algo", fake_register_algo)
    puffer_sac.register()

    assert len(calls) == 1
    name, config_cls, train_fn, backend = calls[0]
    assert name == "sac"
    assert backend == "pufferlib"
    assert config_cls is puffer_sac.SACConfig
    assert train_fn is puffer_sac.train_sac_puffer


def test_puffer_sac_train_delegates_to_impl(monkeypatch):
    from rl.pufferlib.sac import engine

    sentinel = puffer_sac.TrainResult(
        best_return=1.0,
        last_eval_return=0.5,
        last_heldout_return=0.4,
        num_steps=12,
    )
    monkeypatch.setattr(engine, "train_sac_puffer_impl", lambda _cfg: sentinel)

    out = engine.train_sac_puffer(puffer_sac.SACConfig())
    assert out is sentinel


def test_runtime_utils_obs_scaler_select_device_and_obs_scale():
    from rl.pufferlib.sac import runtime_utils as ru

    scaler = ru.ObsScaler(
        np.asarray([1.0, -1.0], dtype=np.float32),
        np.asarray([2.0, 4.0], dtype=np.float32),
    )
    out = scaler(torch.as_tensor([[3.0, 3.0]], dtype=torch.float32))
    assert np.allclose(out.detach().cpu().numpy(), np.asarray([[1.0, 1.0]], dtype=np.float32))
    with pytest.raises(RuntimeError, match="dtype"):
        _ = scaler(torch.as_tensor([[3.0, 3.0]], dtype=torch.float64))

    assert ru.select_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="Unsupported device"):
        _ = ru.select_device("bad-device")

    env_no_scale = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=False),
        ensure_spaces=lambda: None,
    )
    lb, width = ru.obs_scale_from_env(env_no_scale)
    assert lb is None and width is None

    state_space = SimpleNamespace(
        low=np.asarray([-1.0, -2.0], dtype=np.float32),
        high=np.asarray([1.0, 2.0], dtype=np.float32),
        shape=(2,),
    )
    env_scaled = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=True, state_space=state_space),
        ensure_spaces=lambda: None,
    )
    lb2, width2 = ru.obs_scale_from_env(env_scaled)
    assert np.allclose(lb2, state_space.low)
    assert np.allclose(width2, np.asarray([2.0, 4.0], dtype=np.float32))

    bad_state_space = SimpleNamespace(
        low=np.asarray([0.0, 0.0], dtype=np.float32),
        high=np.asarray([1.0, np.inf], dtype=np.float32),
        shape=(2,),
    )
    bad_env = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=True, state_space=bad_state_space),
        ensure_spaces=lambda: None,
    )
    with pytest.raises(ValueError, match="finite"):
        _ = ru.obs_scale_from_env(bad_env)


def test_replay_backend_auto_resolution():
    from rl.core.replay import resolve_replay_backend

    assert resolve_replay_backend("auto", device=torch.device("cpu"), platform_name="darwin") == "numpy"
    assert resolve_replay_backend("auto", device=torch.device("cuda"), platform_name="linux") == "torchrl"
    assert resolve_replay_backend("auto", device=torch.device("cpu"), platform_name="linux") == "numpy"
    assert resolve_replay_backend("numpy", device=torch.device("cuda"), platform_name="linux") == "numpy"
    with pytest.raises(ValueError, match="Unsupported replay backend"):
        _ = resolve_replay_backend("bad-backend", device=torch.device("cpu"), platform_name="linux")


def test_pixel_utils_formats_images():
    from rl.pufferlib.sac.pixel_utils import ensure_pixel_obs_format

    hwc_u8 = torch.randint(0, 255, size=(84, 84, 3), dtype=torch.uint8)
    chw = ensure_pixel_obs_format(hwc_u8, channels=3, size=84, scale_float_255=True)
    assert chw.shape == (3, 84, 84)
    assert chw.dtype == torch.float32

    stacked_gray = torch.randint(0, 255, size=(2, 84, 84, 1), dtype=torch.uint8)
    out = ensure_pixel_obs_format(stacked_gray, channels=4, size=84)
    assert out.shape == (2, 4, 84, 84)

    with pytest.raises(ValueError, match="ndim"):
        _ = ensure_pixel_obs_format(torch.zeros((2,)), channels=3)


def test_env_utils_helpers_and_builders(monkeypatch):
    from rl.pufferlib.sac import env_utils

    env_utils.seed_everything(123)
    a = float(np.random.rand())
    env_utils.seed_everything(123)
    b = float(np.random.rand())
    assert a == b

    assert env_utils.resolve_device("cpu").type == "cpu"
    action = np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32)
    mapped = env_utils.to_env_action(
        action,
        low=np.asarray([-2.0, -2.0, -2.0], dtype=np.float32),
        high=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    )
    assert np.allclose(mapped, np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float32))

    cfg_vec = puffer_sac.SACConfig(from_pixels=False, backbone_name="mlp")
    spec_vec = env_utils.infer_observation_spec(cfg_vec, np.zeros((2, 5), dtype=np.float32))
    assert spec_vec.mode == "vector"
    assert spec_vec.vector_dim == 5

    cfg_px = puffer_sac.SACConfig(from_pixels=True, backbone_name="mlp", framestack=4)
    spec_px = env_utils.infer_observation_spec(cfg_px, np.zeros((2, 84, 84, 4), dtype=np.uint8))
    assert spec_px.mode == "pixels"
    assert spec_px.channels == 4
    assert env_utils.resolve_backbone_name(cfg_px, spec_px) == "nature_cnn_atari"

    vec_in = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    vec_out = env_utils.prepare_obs_np(vec_in, obs_spec=spec_vec)
    assert vec_out.shape == (2, 6)

    px_spec = env_utils.ObservationSpec(mode="pixels", raw_shape=(84, 84, 3), channels=3, image_size=84)
    px_out = env_utils.prepare_obs_np(np.zeros((84, 84, 3), dtype=np.uint8), obs_spec=px_spec)
    assert px_out.shape == (1, 3, 84, 84)

    with pytest.raises(ValueError, match="Observation must include"):
        _ = env_utils.infer_observation_spec(cfg_vec, np.asarray(1.0))

    fake_env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=True),
    )
    monkeypatch.setattr(
        env_utils,
        "build_continuous_gym_env_setup",
        lambda **_kwargs: SimpleNamespace(
            env_conf=fake_env_conf,
            problem_seed=11,
            noise_seed_0=22,
            obs_lb=np.asarray([-1.0, -1.0], dtype=np.float32),
            obs_width=np.asarray([2.0, 2.0], dtype=np.float32),
            act_dim=2,
            action_low=np.asarray([-2.0, -1.0], dtype=np.float32),
            action_high=np.asarray([2.0, 1.0], dtype=np.float32),
        ),
    )

    built = env_utils.build_env_setup(puffer_sac.SACConfig(env_tag="pend", seed=7))
    assert built.problem_seed == 11
    assert built.noise_seed_0 == 22
    assert built.act_dim == 2
    assert built.obs_lb is not None and built.obs_width is not None

    captured = {}

    def _fake_make_vector_env_shared(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return "vec-env"

    monkeypatch.setattr(env_utils, "_make_vector_env_shared", _fake_make_vector_env_shared)
    out_vec = env_utils.make_vector_env(puffer_sac.SACConfig())
    assert out_vec == "vec-env"
    assert "import_pufferlib_modules_fn" in captured["kwargs"]


def _make_small_modules():
    from rl.pufferlib.sac import env_utils, model_utils

    env_setup = env_utils.EnvSetup(
        env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
        problem_seed=0,
        noise_seed_0=0,
        obs_lb=None,
        obs_width=None,
        act_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    obs_spec = env_utils.ObservationSpec(mode="vector", raw_shape=(3,), vector_dim=3)
    cfg = puffer_sac.SACConfig(
        backbone_hidden_sizes=(8,),
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        batch_size=4,
        target_entropy=-2.0,
    )
    modules = model_utils.build_modules(cfg, env_setup, obs_spec, device=torch.device("cpu"))
    optimizers = model_utils.build_optimizers(cfg, modules)
    return cfg, env_setup, obs_spec, modules, optimizers


def test_replay_and_model_utils_update_paths():
    from rl.pufferlib.sac import model_utils
    from rl.pufferlib.sac.replay import ReplayBuffer
    from rl.pufferlib.sac.runtime_utils import ObsScaler

    cfg, _env_setup, _obs_spec, modules, optimizers = _make_small_modules()
    obs = torch.randn(4, 3)

    action, log_prob = modules.actor.sample(obs, deterministic=False)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    action_det, log_prob_det = modules.actor.sample(obs, deterministic=True)
    assert action_det.shape == (4, 2)
    assert torch.all(log_prob_det == 0.0)

    q_vals = modules.q1(obs, action)
    assert q_vals.shape == (4,)

    pixel_q = model_utils.QNetPixel(
        obs_encoder=nn.Identity(),
        head=nn.Linear(6, 1),
        obs_scaler=ObsScaler(None, None),
    )
    px_out = pixel_q(torch.randn(4, 4), torch.randn(4, 2))
    assert px_out.shape == (4,)

    alpha_val = model_utils.alpha(modules)
    assert float(alpha_val.item()) > 0.0

    snap = model_utils.capture_actor_state(modules)
    for p in modules.actor_head.parameters():
        p.data.add_(1.0)
    model_utils.restore_actor_state(modules, snap)
    with model_utils.use_actor_state(modules, snap):
        _ = modules.actor.act(obs)

    replay = ReplayBuffer(obs_shape=(3,), act_dim=2, capacity=32)
    for _ in range(3):
        replay.add_batch(
            obs=np.random.randn(4, 3).astype(np.float32),
            act=np.random.randn(4, 2).astype(np.float32),
            rew=np.random.randn(4).astype(np.float32),
            nxt=np.random.randn(4, 3).astype(np.float32),
            done=np.random.randint(0, 2, size=(4,), dtype=np.int32).astype(np.float32),
        )
    assert replay.size > 0
    sample = replay.sample(batch_size=4, device=torch.device("cpu"))
    assert len(sample) == 5
    assert sample[0].shape == (4, 3)
    saved = replay.state_dict()
    replay_loaded = ReplayBuffer(obs_shape=(3,), act_dim=2, capacity=32)
    replay_loaded.load_state_dict(saved)
    assert replay_loaded.ptr == replay.ptr
    assert replay_loaded.size == replay.size
    assert np.array_equal(replay_loaded.obs, replay.obs)
    assert np.array_equal(replay_loaded.nxt, replay.nxt)
    assert np.array_equal(replay_loaded.act, replay.act)
    assert np.array_equal(replay_loaded.rew, replay.rew)
    assert np.array_equal(replay_loaded.done, replay.done)

    from rl.core.replay import make_replay_buffer

    replay_torchrl = make_replay_buffer(
        obs_shape=(3,),
        act_dim=2,
        capacity=32,
        backend="torchrl",
    )
    replay_torchrl.add_batch(
        obs=np.random.randn(4, 3).astype(np.float32),
        act=np.random.randn(4, 2).astype(np.float32),
        rew=np.random.randn(4).astype(np.float32),
        nxt=np.random.randn(4, 3).astype(np.float32),
        done=np.random.randint(0, 2, size=(4,), dtype=np.int32).astype(np.float32),
    )
    sampled_torchrl = replay_torchrl.sample(batch_size=2, device=torch.device("cpu"))
    assert len(sampled_torchrl) == 5
    assert sampled_torchrl[0].shape[1:] == (3,)
    torchrl_state = replay_torchrl.state_dict()
    replay_torchrl_loaded = make_replay_buffer(
        obs_shape=(3,),
        act_dim=2,
        capacity=32,
        backend="torchrl",
    )
    replay_torchrl_loaded.load_state_dict(torchrl_state)
    assert replay_torchrl_loaded.size == replay_torchrl.size

    before = [p.detach().clone() for p in modules.q1_target.parameters()]
    actor_loss, critic_loss, alpha_loss = model_utils.sac_update(
        cfg,
        modules,
        optimizers,
        replay,
        device=torch.device("cpu"),
    )
    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert isinstance(alpha_loss, float)
    after = list(modules.q1_target.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))


def test_eval_utils_paths(monkeypatch, tmp_path: Path):
    from rl.pufferlib.sac import eval_utils

    cfg, env_setup, obs_spec, modules, _optimizers = _make_small_modules()
    state = eval_utils.TrainState(start_time=float(time.time()) - 1.0)

    class _Actor:
        @staticmethod
        def act(obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0], 2), dtype=torch.float32)

    policy = eval_utils.SacEvalPolicy(SimpleNamespace(actor=_Actor()), obs_spec, device=torch.device("cpu"))
    out = policy(np.zeros((3,), dtype=np.float32))
    assert out.shape == (2,)

    monkeypatch.setattr(
        eval_utils,
        "collect_denoised_trajectory",
        lambda _env_conf, _policy, **_kwargs: (SimpleNamespace(rreturn=3.5), 0),
    )
    eval_return = eval_utils.evaluate_actor(
        cfg,
        env_setup,
        SimpleNamespace(actor=_Actor()),
        obs_spec,
        device=torch.device("cpu"),
        eval_seed=0,
    )
    assert eval_return == 3.5

    cfg_no_heldout = puffer_sac.SACConfig(num_denoise_passive_eval=None)
    assert (
        eval_utils.evaluate_heldout_if_enabled(
            cfg_no_heldout,
            env_setup,
            SimpleNamespace(actor=_Actor()),
            obs_spec,
            device=torch.device("cpu"),
            heldout_i_noise=0,
        )
        is None
    )

    monkeypatch.setattr(eval_utils, "evaluate_for_best", lambda *_args, **_kwargs: 4.2)
    heldout = eval_utils.evaluate_heldout_if_enabled(
        puffer_sac.SACConfig(num_denoise_passive_eval=2),
        env_setup,
        SimpleNamespace(actor=_Actor()),
        obs_spec,
        device=torch.device("cpu"),
        heldout_i_noise=7,
    )
    assert heldout == 4.2

    assert eval_utils.due_mark(0, 10, 0) is None
    assert eval_utils.due_mark(10, 10, 0) == 1
    assert eval_utils.due_mark(10, 0, 0) is None

    metric_calls = []
    monkeypatch.setattr(eval_utils.rl_logger, "append_metrics", lambda path, record: metric_calls.append((path, record)))
    state.last_eval_return = 1.25
    state.last_heldout_return = 0.75
    state.best_return = 1.25
    eval_utils.append_eval_metric(tmp_path / "metrics.jsonl", state, step=10)
    assert len(metric_calls) == 1
    assert metric_calls[0][1]["step"] == 10

    log_calls = []
    monkeypatch.setattr(eval_utils.rl_logger, "log_eval_iteration", lambda **kwargs: log_calls.append(kwargs))
    log_cfg = puffer_sac.SACConfig(log_interval_steps=5)
    eval_utils.log_if_due(log_cfg, state, step=5, frames_per_batch=8)
    assert len(log_calls) == 1
    assert state.log_mark == 1

    monkeypatch.setattr(eval_utils, "build_eval_plan", lambda **_kwargs: SimpleNamespace(eval_seed=1, heldout_i_noise=2))
    monkeypatch.setattr(eval_utils, "evaluate_actor", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(eval_utils, "evaluate_heldout_if_enabled", lambda *_args, **_kwargs: 1.5)
    eval_cfg = puffer_sac.SACConfig(eval_interval_steps=10, num_denoise_passive_eval=1)
    state2 = eval_utils.TrainState(global_step=10, start_time=float(time.time()) - 1.0)
    eval_utils.maybe_eval(eval_cfg, env_setup, modules, obs_spec, state2, device=torch.device("cpu"))
    assert state2.eval_mark == 1
    assert state2.best_return == 2.0
    assert state2.best_actor_state is not None
    assert state2.last_heldout_return == 1.5

    state2.global_step = 20
    monkeypatch.setattr(eval_utils, "evaluate_actor", lambda *_args, **_kwargs: 1.0)
    eval_utils.maybe_eval(eval_cfg, env_setup, modules, obs_spec, state2, device=torch.device("cpu"))
    assert state2.best_return == 2.0

    video_calls = []
    monkeypatch.setattr(
        "common.video.render_policy_videos",
        lambda *args, **kwargs: video_calls.append((args, kwargs)),
    )
    video_cfg = puffer_sac.SACConfig(
        exp_dir=str(tmp_path / "exp"),
        video_enable=True,
        video_num_episodes=3,
        video_num_video_episodes=1,
        video_seed_base=None,
    )
    eval_utils.render_videos_if_enabled(video_cfg, env_setup, modules, obs_spec, device=torch.device("cpu"))
    assert len(video_calls) == 1
    assert video_calls[0][1]["seed_base"] == env_setup.problem_seed + 10000


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


def test_train_sac_puffer_impl_smoke_with_patched_loop(monkeypatch, tmp_path: Path):
    from rl.pufferlib.sac import engine

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
        engine,
        "_init_run_artifacts",
        lambda _cfg: (tmp_path / "exp", tmp_path / "exp" / "metrics.jsonl", SimpleNamespace(save_both=lambda *_args, **_kwargs: None)),
    )
    monkeypatch.setattr(engine, "_init_runtime", lambda _cfg: (env_setup, torch.device("cpu")))
    monkeypatch.setattr(engine, "make_vector_env", lambda _cfg: _FakeVecEnv(num_envs=1, obs_dim=3))
    monkeypatch.setattr(engine, "infer_observation_spec", lambda _cfg, _obs: obs_spec)
    monkeypatch.setattr(engine, "prepare_obs_np", lambda obs_np, **_kwargs: np.asarray(obs_np, dtype=np.float32))
    monkeypatch.setattr(
        engine,
        "_build_training_components",
        lambda *_args, **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            engine.TrainState(start_time=float(time.time()) - 1.0),
        ),
    )
    monkeypatch.setattr(engine, "_log_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("rl.logger.log_run_footer", lambda **_kwargs: None)
    monkeypatch.setattr(engine, "render_videos_if_enabled", lambda *_args, **_kwargs: None)

    def _fake_train_loop(_config, _env_setup, _modules, _optimizers, _replay, state, _obs_spec, _obs_batch, _envs, **_kwargs):
        state.global_step = int(_config.total_timesteps)
        state.best_return = 1.0
        state.last_eval_return = 0.5
        state.last_heldout_return = 0.25

    monkeypatch.setattr(engine, "_train_loop", _fake_train_loop)

    out = puffer_sac.train_sac_puffer(cfg)
    assert out.num_steps == 8
    assert out.best_return == 1.0
