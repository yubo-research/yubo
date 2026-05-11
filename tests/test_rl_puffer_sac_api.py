def test_puffer_sac_config_from_dict_converts_hidden_sizes():
    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")

    cfg = puffer_sac.SACConfig.from_dict(
        {
            "env_tag": "cheetah",
            "policy_tag": "mlp-32-16",
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


def test_puffer_sac_config_from_dict_uses_env_defaults():
    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")

    cfg = puffer_sac.SACConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})
    assert cfg.backbone_hidden_sizes == (256, 256)


def test_puffer_sac_register_delegates_to_registry(monkeypatch):
    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")

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
    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")
    sac_train_run = import_dotted("rl", "pufferlib", "sac", "sac_puffer_train_run")
    engine = import_dotted("rl", "pufferlib", "sac", "engine")

    sentinel = puffer_sac.TrainResult(
        best_return=1.0,
        last_eval_return=0.5,
        last_heldout_return=0.4,
        num_steps=12,
    )
    monkeypatch.setattr(sac_train_run, "train_sac_puffer_impl", lambda _cfg: sentinel)

    out = engine.train_sac_puffer(puffer_sac.SACConfig())
    assert out is sentinel


def test_runtime_utils_obs_scaler_select_device_and_obs_scale():
    from types import SimpleNamespace

    import numpy as np
    import pytest
    import torch

    from testing_support.dyn_import import import_dotted

    ru = import_dotted("rl", "pufferlib", "sac", "runtime_utils")

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
    import pytest
    import torch

    from testing_support.dyn_import import import_dotted

    rr = import_dotted("rl", "core", "replay")
    resolve_replay_backend = rr.resolve_replay_backend

    assert resolve_replay_backend("auto", device=torch.device("cpu"), platform_name="darwin") == "numpy"
    assert resolve_replay_backend("auto", device=torch.device("cuda"), platform_name="linux") == "torchrl"
    assert resolve_replay_backend("auto", device=torch.device("cpu"), platform_name="linux") == "numpy"
    assert resolve_replay_backend("numpy", device=torch.device("cuda"), platform_name="linux") == "numpy"
    with pytest.raises(ValueError, match="Unsupported replay backend"):
        _ = resolve_replay_backend("bad-backend", device=torch.device("cpu"), platform_name="linux")


def test_pixel_utils_formats_images():
    import pytest
    import torch

    from testing_support.dyn_import import import_dotted

    pu = import_dotted("rl", "pufferlib", "sac", "pixel_utils")
    ensure_pixel_obs_format = pu.ensure_pixel_obs_format

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
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")
    env_utils = import_dotted("rl", "pufferlib", "sac", "env_utils")
    offpolicy_env_utils = import_dotted("rl", "pufferlib", "offpolicy", "env_utils")

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
        offpolicy_env_utils,
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
