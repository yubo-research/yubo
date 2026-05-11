def test_replay_and_model_utils_update_paths():
    from rl_puffer_sac_replay_bundle import build_replay_sac_bundle

    b = build_replay_sac_bundle()
    obs = b.torch.randn(4, 3)

    action, log_prob = b.modules.actor.sample(obs, deterministic=False)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    action_det, log_prob_det = b.modules.actor.sample(obs, deterministic=True)
    assert action_det.shape == (4, 2)
    assert b.torch.all(log_prob_det == 0.0)

    q_vals = b.modules.q1(obs, action)
    assert q_vals.shape == (4,)

    pixel_q = b.model_utils.QNetPixel(
        obs_encoder=b.nn.Identity(),
        head=b.nn.Linear(6, 1),
        obs_scaler=b.ObsScaler(None, None),
    )
    px_out = pixel_q(b.torch.randn(4, 4), b.torch.randn(4, 2))
    assert px_out.shape == (4,)

    alpha_val = b.model_utils.alpha(b.modules)
    assert float(alpha_val.item()) > 0.0

    snap = b.model_utils.capture_actor_state(b.modules)
    for p in b.modules.actor_head.parameters():
        p.data.add_(1.0)
    b.model_utils.restore_actor_state(b.modules, snap)
    with b.model_utils.use_actor_state(b.modules, snap):
        _ = b.modules.actor.act(obs)

    for _ in range(3):
        b.replay.add_batch(
            obs=b.np.random.randn(4, 3).astype(b.np.float32),
            act=b.np.random.randn(4, 2).astype(b.np.float32),
            rew=b.np.random.randn(4).astype(b.np.float32),
            nxt=b.np.random.randn(4, 3).astype(b.np.float32),
            done=b.np.random.randint(0, 2, size=(4,), dtype=b.np.int32).astype(b.np.float32),
        )
    assert b.replay.size > 0
    sample = b.replay.sample(batch_size=4, device=b.torch.device("cpu"))
    assert len(sample) == 5
    assert sample[0].shape == (4, 3)
    saved = b.replay.state_dict()
    replay_loaded = b.ReplayBuffer(obs_shape=(3,), act_dim=2, capacity=32)
    replay_loaded.load_state_dict(saved)
    assert replay_loaded.ptr == b.replay.ptr
    assert replay_loaded.size == b.replay.size
    assert b.np.array_equal(replay_loaded.obs, b.replay.obs)
    assert b.np.array_equal(replay_loaded.nxt, b.replay.nxt)
    assert b.np.array_equal(replay_loaded.act, b.replay.act)
    assert b.np.array_equal(replay_loaded.rew, b.replay.rew)
    assert b.np.array_equal(replay_loaded.done, b.replay.done)

    replay_torchrl = b.make_replay_buffer(
        obs_shape=(3,),
        act_dim=2,
        capacity=32,
        backend="torchrl",
    )
    replay_torchrl.add_batch(
        obs=b.np.random.randn(4, 3).astype(b.np.float32),
        act=b.np.random.randn(4, 2).astype(b.np.float32),
        rew=b.np.random.randn(4).astype(b.np.float32),
        nxt=b.np.random.randn(4, 3).astype(b.np.float32),
        done=b.np.random.randint(0, 2, size=(4,), dtype=b.np.int32).astype(b.np.float32),
    )
    sampled_torchrl = replay_torchrl.sample(batch_size=2, device=b.torch.device("cpu"))
    assert len(sampled_torchrl) == 5
    assert sampled_torchrl[0].shape[1:] == (3,)
    torchrl_state = replay_torchrl.state_dict()
    replay_torchrl_loaded = b.make_replay_buffer(
        obs_shape=(3,),
        act_dim=2,
        capacity=32,
        backend="torchrl",
    )
    replay_torchrl_loaded.load_state_dict(torchrl_state)
    assert replay_torchrl_loaded.size == replay_torchrl.size

    before = [p.detach().clone() for p in b.modules.q1_target.parameters()]
    actor_loss, critic_loss, alpha_loss = b.model_utils.sac_update(
        b.cfg,
        b.modules,
        b.optimizers,
        b.replay,
        device=b.torch.device("cpu"),
    )
    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert isinstance(alpha_loss, float)
    after = list(b.modules.q1_target.parameters())
    assert any(not b.torch.allclose(b_, a) for b_, a in zip(before, after, strict=True))
