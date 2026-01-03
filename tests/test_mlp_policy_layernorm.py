def test_bw_mlp_uses_layernorm_with_affine_and_episode_reset_only_resets_rnn_state():
    from types import SimpleNamespace

    import torch

    from problems.mlp_policy import MLPPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
    )

    p = MLPPolicy(env_conf, hidden_sizes=(), use_layer_norm=True, rnn_hidden_size=8)
    assert p.in_norm is not None
    assert isinstance(p.in_norm, torch.nn.LayerNorm)
    assert p.in_norm.elementwise_affine is True
    assert p.in_norm.weight.requires_grad is True
    assert p.in_norm.bias.requires_grad is True

    p.reset_state()
    h0 = p._h.clone()
    p._h[:] = 1.0
    p.reset_state()
    assert torch.allclose(p._h, h0)
