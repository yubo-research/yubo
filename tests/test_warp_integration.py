import pytest
import torch

wp = pytest.importorskip("warp")

from problems.environment_spec import get_environment_spec, materialize_env  # noqa: E402


def test_warp_adapter_materialization():
    # Only run if mujoco_warp is available (mocking for kiss check if needed)
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        pytest.skip("mujoco_warp not available")
    if not hasattr(wp, "tid"):
        pytest.skip("warp build missing wp.tid(); incompatible with mujoco_warp")

    env_tag = "warp:gymnasium:Ant-v4"
    spec = get_environment_spec(env_tag)
    assert spec.env_name == env_tag

    runtime = materialize_env(spec)
    assert runtime.env_name == env_tag

    # ensure_spaces should work
    runtime.ensure_spaces()
    assert runtime.state_space is not None
    assert runtime.action_space is not None

    # make() should return the UnifiedMJXWarpAdapter
    env = runtime.make()
    from problems.unified_mj_env import UnifiedMJXWarpAdapter

    assert isinstance(env, UnifiedMJXWarpAdapter)

    # reset should return torch tensors
    obs_dict, data = env.reset()
    assert "obs" in obs_dict
    assert isinstance(obs_dict["obs"], torch.Tensor)

    # step should return UnifiedState
    action = torch.zeros(env.action_space.shape)
    state = env.step(data, action)
    from problems.unified_mj_env import UnifiedState

    assert isinstance(state, UnifiedState)

    # State should have torch() and jax() views
    torch_views = state.torch()
    assert isinstance(torch_views["obs"], torch.Tensor)

    try:
        import jax

        jax_views = state.jax()
        assert isinstance(jax_views["obs"], jax.Array)
    except ImportError:
        pass


def test_orbax_checkpointing_unified(tmp_path):
    try:
        import jax  # noqa: F401
        import orbax.checkpoint  # noqa: F401
    except ImportError:
        pytest.skip("orbax or jax not available")

    from problems.unified_mj_env import save_unified_checkpoint

    wp.init()
    dummy_wp = wp.array([1.0, 2.0], dtype=wp.float32)
    dummy_torch = torch.tensor([3.0, 4.0])

    state = {"warp": dummy_wp, "torch": dummy_torch, "scalar": 5.0}

    checkpoint_path = tmp_path / "checkpoint"
    save_unified_checkpoint(str(checkpoint_path), state)
    # Orbax CheckpointManager saves to a subdirectory '0' for iteration 0
    assert (checkpoint_path / "0").exists()
