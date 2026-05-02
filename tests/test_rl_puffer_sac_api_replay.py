def test_replay_and_model_utils_update_paths():
    from types import SimpleNamespace

    import numpy as np
    import torch
    import torch.nn as nn

    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")
    env_utils = import_dotted("rl", "pufferlib", "sac", "env_utils")
    model_utils = import_dotted("rl", "pufferlib", "sac", "model_utils")
    replay_mod = import_dotted("rl", "pufferlib", "sac", "replay")
    ReplayBuffer = replay_mod.ReplayBuffer
    runtime_utils = import_dotted("rl", "pufferlib", "sac", "runtime_utils")
    ObsScaler = runtime_utils.ObsScaler

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

    core_replay = import_dotted("rl", "core", "replay")
    make_replay_buffer = core_replay.make_replay_buffer

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
