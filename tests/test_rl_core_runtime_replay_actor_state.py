from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from rl.core import actor_state, replay, runtime


def test_runtime_paths_and_obs_scaler():
    scaler_identity = runtime.ObsScaler(None, None)
    obs = torch.ones((2, 3), dtype=torch.float32)
    assert torch.allclose(scaler_identity(obs), obs)

    scaler = runtime.ObsScaler(np.asarray([1.0, 2.0], dtype=np.float32), np.asarray([2.0, 4.0], dtype=np.float32))
    out = scaler(torch.as_tensor([[3.0, 6.0]], dtype=torch.float32))
    assert np.allclose(out.detach().cpu().numpy(), np.asarray([[1.0, 1.0]], dtype=np.float32))
    with pytest.raises(RuntimeError, match="dtype"):
        _ = scaler(torch.as_tensor([[3.0, 6.0]], dtype=torch.float64))

    assert isinstance(runtime.mps_is_available(), bool)
    assert runtime.select_device("cpu").type == "cpu"
    assert runtime.select_device("auto", cuda_is_available_fn=lambda: False, mps_is_available_fn=lambda: False).type == "cpu"
    assert runtime.select_device("auto", cuda_is_available_fn=lambda: True, mps_is_available_fn=lambda: False).type == "cuda"
    assert runtime.select_device("auto", cuda_is_available_fn=lambda: False, mps_is_available_fn=lambda: True).type == "mps"
    with pytest.raises(ValueError, match="Unsupported device"):
        _ = runtime.select_device("bad")
    with pytest.raises(ValueError, match="CUDA is not available"):
        _ = runtime.select_device("cuda", cuda_is_available_fn=lambda: False, mps_is_available_fn=lambda: False)
    with pytest.raises(ValueError, match="MPS is not available"):
        _ = runtime.select_device("mps", cuda_is_available_fn=lambda: False, mps_is_available_fn=lambda: False)

    seeded: list[int] = []
    runtime.seed_everything(123, cuda_is_available_fn=lambda: True, cuda_manual_seed_all_fn=lambda s: seeded.append(int(s)))
    assert seeded == [123]

    env_no_transform = SimpleNamespace(gym_conf=SimpleNamespace(transform_state=False), ensure_spaces=lambda: None)
    lb, width = runtime.obs_scale_from_env(env_no_transform)
    assert lb is None and width is None

    space = SimpleNamespace(low=np.asarray([-1.0, -2.0], dtype=np.float32), high=np.asarray([1.0, 2.0], dtype=np.float32), shape=(2,))
    env_transform = SimpleNamespace(gym_conf=SimpleNamespace(transform_state=True, state_space=space), ensure_spaces=lambda: None)
    lb2, width2 = runtime.obs_scale_from_env(env_transform)
    assert np.allclose(lb2, np.asarray([-1.0, -2.0], dtype=np.float32))
    assert np.allclose(width2, np.asarray([2.0, 4.0], dtype=np.float32))

    bad_space = SimpleNamespace(low=np.asarray([0.0, np.inf], dtype=np.float32), high=np.asarray([1.0, 2.0], dtype=np.float32), shape=(2,))
    bad_env = SimpleNamespace(gym_conf=SimpleNamespace(transform_state=True, state_space=bad_space), ensure_spaces=lambda: None)
    with pytest.raises(ValueError, match="finite"):
        _ = runtime.obs_scale_from_env(bad_env)

    kwargs = runtime.collector_device_kwargs(torch.device("cpu"))
    assert kwargs["env_device"].type == "cpu"
    assert kwargs["policy_device"].type == "cpu"
    prev = bool(getattr(torch.distributions.Distribution, "_validate_args", True))
    with runtime.temporary_distribution_validate_args(False):
        assert bool(getattr(torch.distributions.Distribution, "_validate_args", True)) is False
    assert bool(getattr(torch.distributions.Distribution, "_validate_args", True)) == prev


def test_replay_buffers_and_backend_resolution():
    rb = replay.NumpyReplayBuffer(obs_shape=(3,), act_dim=2, capacity=8)
    rb.add_batch(
        obs=np.random.randn(4, 3).astype(np.float32),
        act=np.random.randn(4, 2).astype(np.float32),
        rew=np.random.randn(4).astype(np.float32),
        nxt=np.random.randn(4, 3).astype(np.float32),
        done=np.asarray([0, 1, 0, 1], dtype=np.float32),
    )
    sampled = rb.sample(batch_size=2, device=torch.device("cpu"))
    assert len(sampled) == 5
    state = rb.state_dict()
    rb2 = replay.NumpyReplayBuffer(obs_shape=(3,), act_dim=2, capacity=8)
    rb2.load_state_dict(state)
    assert rb2.size == rb.size
    bad = dict(state)
    bad["obs"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        rb2.load_state_dict(bad)

    rb_torch = replay.TorchRLReplayBuffer(obs_shape=(3,), act_dim=2, capacity=8)
    rb_torch.add_batch(
        obs=np.random.randn(4, 3).astype(np.float32),
        act=np.random.randn(4, 2).astype(np.float32),
        rew=np.random.randn(4).astype(np.float32),
        nxt=np.random.randn(4, 3).astype(np.float32),
        done=np.asarray([0, 1, 0, 1], dtype=np.float32),
    )
    sampled_torch = rb_torch.sample(batch_size=2, device=torch.device("cpu"))
    assert len(sampled_torch) == 5
    state_torch = rb_torch.state_dict()
    rb_torch2 = replay.TorchRLReplayBuffer(obs_shape=(3,), act_dim=2, capacity=8)
    rb_torch2.load_state_dict(state_torch)
    assert rb_torch2.size == rb_torch.size

    assert isinstance(replay.make_replay_buffer(obs_shape=(3,), act_dim=2, capacity=8, backend="numpy"), replay.NumpyReplayBuffer)
    assert isinstance(replay.make_replay_buffer(obs_shape=(3,), act_dim=2, capacity=8, backend="torchrl"), replay.TorchRLReplayBuffer)
    with pytest.raises(ValueError, match="Unsupported replay backend"):
        _ = replay.make_replay_buffer(obs_shape=(3,), act_dim=2, capacity=8, backend="bad")
    assert replay.resolve_replay_backend("auto", device=torch.device("cpu"), platform_name="darwin") == "numpy"
    assert replay.resolve_replay_backend("auto", device=torch.device("cuda"), platform_name="linux") == "torchrl"
    with pytest.raises(ValueError, match="Unsupported replay backend"):
        _ = replay.resolve_replay_backend("bad", device=torch.device("cpu"))


def test_actor_state_snapshot_paths():
    actor_backbone = nn.Linear(3, 4)
    actor_head = nn.Linear(4, 2)
    log_std = nn.Parameter(torch.zeros(2))

    snap_tensor = actor_state.capture_backbone_head_snapshot(actor_backbone, actor_head, log_std=log_std, log_std_format="tensor")
    assert "backbone" in snap_tensor and "head" in snap_tensor and "log_std" in snap_tensor
    snap_numpy = actor_state.capture_backbone_head_snapshot(
        actor_backbone,
        actor_head,
        log_std=log_std,
        log_std_to_cpu=True,
        log_std_format="numpy",
    )
    assert isinstance(snap_numpy["log_std"], np.ndarray)
    with pytest.raises(ValueError, match="log_std_format"):
        _ = actor_state.capture_backbone_head_snapshot(actor_backbone, actor_head, log_std=log_std, log_std_format="bad")

    ppo_snap = actor_state.capture_ppo_actor_snapshot(actor_backbone, actor_head, log_std=log_std)
    assert "log_std" in ppo_snap

    with torch.no_grad():
        for p in actor_backbone.parameters():
            p.add_(1.0)
    actor_state.restore_backbone_head_snapshot(actor_backbone, actor_head, snap_tensor, log_std=log_std, device=torch.device("cpu"))

    before = {k: v.detach().clone() for k, v in actor_backbone.state_dict().items()}
    with actor_state.use_backbone_head_snapshot(
        actor_backbone,
        actor_head,
        snap_tensor,
        log_std=log_std,
        device=torch.device("cpu"),
        state_to_cpu=True,
        log_std_to_cpu=True,
        log_std_format="tensor",
    ):
        with torch.no_grad():
            for p in actor_backbone.parameters():
                p.mul_(0.0)
    after = actor_backbone.state_dict()
    for key in before:
        assert torch.allclose(after[key], before[key])

    payload_rng = actor_state.rng_state_payload()
    assert "rng_torch" in payload_rng and "rng_numpy" in payload_rng and "rng_cuda" in payload_rng

    payload = actor_state.build_ppo_checkpoint_payload(
        iteration=3,
        global_step=128,
        actor_snapshot=ppo_snap,
        critic_backbone={"w": torch.tensor([1.0])},
        critic_head={"w": torch.tensor([2.0])},
        optimizer={"state": {}},
        best_actor_state={"k": 1},
        best_return=2.5,
        last_eval_return=2.0,
        last_heldout_return=1.5,
        extra_payload={"x": 7},
    )
    assert payload["iteration"] == 3
    assert payload["global_step"] == 128
    assert payload["x"] == 7
