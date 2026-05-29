import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from problems.isaaclab_config import disable_command_debug_visualizers
from problems.isaaclab_env_adapters import (
    IsaacLabGymEnvAdapter,
    IsaacLabSession,
    IsaacLabVectorEnvAdapter,
    get_isaaclab_session,
    is_isaaclab_env_tag,
    isaaclab_default_launcher_kwargs,
    isaaclab_video_launcher_kwargs,
    list_isaaclab_tasks,
    main,
    make_isaaclab_env,
    make_raw_isaaclab_env,
    parse_isaaclab_task_id,
    resolve_isaaclab_env_spaces,
)


class _FakeIsaacEnv:
    def __init__(self):
        from gymnasium import spaces

        self.device = "cpu"
        self.single_observation_space = spaces.Dict({"policy": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)})
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.closed = False
        self.last_action = None

    def reset(self, *args, **kwargs):
        return {"policy": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}, {
            "reset": True,
            "kwargs": kwargs,
        }

    def step(self, action):
        self.last_action = action
        return (
            {"observation": np.array([[4.0, 5.0, 6.0]], dtype=np.float32)},
            np.array([1.25], dtype=np.float32),
            np.array([False]),
            np.array([True]),
            {"step": True},
        )

    def render(self, *args, **kwargs):
        return {"frame": True, "kwargs": kwargs}

    def close(self):
        self.closed = True


class _FakeGym:
    def __init__(self):
        self.registry = {
            "cartpole": SimpleNamespace(id="Isaac-Cartpole-v0"),
            "ant": SimpleNamespace(id="Isaac-Ant-v0"),
            "other": SimpleNamespace(id="CartPole-v1"),
        }
        self.calls = []

    def make(self, task_id, **kwargs):
        print("        Environment device    : cuda:0")
        print("        Number of environments: 1")
        os.write(2, b"native setup detail\n")
        self.calls.append((task_id, kwargs))
        return _FakeIsaacEnv()


def test_isaaclab_tag_parsing():
    assert is_isaaclab_env_tag("isaaclab:Isaac-Cartpole-v0")
    assert parse_isaaclab_task_id("isaaclab:Isaac-Cartpole-v0") == "Isaac-Cartpole-v0"


def test_get_isaaclab_session_and_list_isaaclab_tasks(monkeypatch):
    import problems.isaaclab_env_adapters as mod

    class _FakeLauncher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.app = SimpleNamespace(name="fake-app")

    fake_gym = _FakeGym()
    monkeypatch.setattr(mod, "_SESSION", None)
    monkeypatch.setattr(mod, "_import_isaac_app_launcher", lambda: _FakeLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)

    session = get_isaaclab_session(headless=False, launcher_kwargs={"experience": "test.kit"})

    assert isinstance(session, IsaacLabSession)
    assert session.app.name == "fake-app"
    assert session.gym is fake_gym
    assert list_isaaclab_tasks(keyword="cart", headless=True) == ["Isaac-Cartpole-v0"]
    assert list_isaaclab_tasks(keyword=None, headless=True) == [
        "Isaac-Ant-v0",
        "Isaac-Cartpole-v0",
    ]


def test_isaaclab_default_launcher_kwargs_provide_headless_kit_args():
    preset = isaaclab_default_launcher_kwargs()
    assert "--no-window" in preset["kit_args"]
    assert "--/renderer/multiGpu/enabled=false" in preset["kit_args"]


def test_isaaclab_replicator_seed_patch_only_swallows_missing_graph(monkeypatch):
    from problems.isaaclab_replicator import patch_replicator_seed_without_graph

    def install_rep(set_global_seed):
        rep = SimpleNamespace(set_global_seed=set_global_seed)
        monkeypatch.setitem(sys.modules, "omni", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "omni.replicator", SimpleNamespace(core=rep))
        monkeypatch.setitem(sys.modules, "omni.replicator.core", rep)
        return rep

    calls = []

    def raise_missing_graph(seed):
        calls.append(seed)
        raise ValueError("Unable to retrieve replicator graph")

    rep = install_rep(raise_missing_graph)
    patch_replicator_seed_without_graph()

    assert rep.set_global_seed(11) is None
    assert calls == [11]
    patched_set_global_seed = rep.set_global_seed
    patch_replicator_seed_without_graph()
    assert rep.set_global_seed is patched_set_global_seed

    def raise_different(_seed):
        raise ValueError("different failure")

    rep = install_rep(raise_different)
    patch_replicator_seed_without_graph()
    with pytest.raises(ValueError, match="different failure"):
        rep.set_global_seed(12)


def test_isaaclab_video_build_problem_reaches_space_resolution(monkeypatch):
    import problems.isaaclab_env_adapters as mod
    from problems.problem import build_problem

    calls = []
    fake_gym = _FakeGym()
    monkeypatch.setattr(mod, "_SPACE_CACHE", {})
    monkeypatch.setattr(
        mod,
        "get_isaaclab_session",
        lambda **kwargs: (calls.append(kwargs) or IsaacLabSession(app=None, gym=fake_gym)),
    )
    monkeypatch.setattr(
        mod,
        "_parse_env_cfg",
        lambda task_id, **kwargs: SimpleNamespace(task_id=task_id, seed=None, **kwargs),
    )

    problem = build_problem(
        "isaaclab:Isaac-Cartpole-v0",
        policy_tag="actor-critic-mlp-32-32",
        problem_seed=3,
        isaaclab_video=True,
    )
    problem.env.ensure_spaces()

    assert calls[0]["launcher_kwargs"] == isaaclab_video_launcher_kwargs()
    assert calls[0]["launcher_kwargs"]["video"] is True
    assert "omni.kit.viewport.utility" in calls[0]["launcher_kwargs"]["kit_args"]
    assert "omni.replicator.core" not in calls[0]["launcher_kwargs"]["kit_args"]


def test_isaaclab_gym_env_adapter_reset_step_render_close():
    env = _FakeIsaacEnv()
    adapter = IsaacLabGymEnvAdapter(env, num_envs=1)

    obs, info = adapter.reset(seed=7)
    next_obs, reward, terminated, truncated, step_info = adapter.step(np.array([0.1, -0.2], dtype=np.float32))
    action = env.last_action.detach().cpu().numpy() if hasattr(env.last_action, "detach") else np.asarray(env.last_action)

    assert obs.tolist() == [1.0, 2.0, 3.0]
    assert info["reset"] is True
    assert next_obs.tolist() == [4.0, 5.0, 6.0]
    assert reward == np.float32(1.25)
    assert terminated is False
    assert truncated is True
    assert step_info == {"step": True}
    assert action.shape == (1, 2)
    assert adapter.render(mode="rgb_array") == {
        "frame": True,
        "kwargs": {"mode": "rgb_array"},
    }

    adapter.close()
    assert env.closed is True


def test_isaaclab_tensor_batch_io_keeps_torch_actions():
    torch = pytest.importorskip("torch")
    from gymnasium import spaces

    from problems.isaaclab_tensor_io import reset_tensor_batch, step_tensor_batch

    env = SimpleNamespace(
        device="cpu",
        single_observation_space=spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        single_action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )
    env.reset = lambda *args, **kwargs: (
        {"policy": torch.ones((2, 3), dtype=torch.float32)},
        {"kwargs": kwargs},
    )

    def _step(action):
        env.last_action = action
        return (
            {"policy": torch.full((2, 3), 2.0, dtype=torch.float32)},
            torch.ones((2,), dtype=torch.float32),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            {},
        )

    env.step = _step
    adapter = IsaacLabVectorEnvAdapter(env, num_envs=2)
    obs, info = reset_tensor_batch(adapter, seed=9)
    next_obs, reward, terminated, truncated, _info = step_tensor_batch(adapter, torch.zeros((2, 2), dtype=torch.float32))

    assert info["kwargs"]["seed"] == 9
    assert tuple(obs.shape) == (2, 3)
    assert tuple(next_obs.shape) == (2, 3)
    assert tuple(reward.shape) == (2,)
    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [False, False]
    assert hasattr(env.last_action, "detach")
    assert tuple(env.last_action.shape) == (2, 2)


def test_make_isaaclab_env_and_resolve_isaaclab_env_spaces(monkeypatch):
    import problems.isaaclab_env_adapters as mod

    fake_gym = _FakeGym()
    monkeypatch.setattr(
        mod,
        "get_isaaclab_session",
        lambda **_kwargs: IsaacLabSession(app=None, gym=fake_gym),
    )
    monkeypatch.setattr(mod, "_parse_env_cfg", lambda task_id, **kwargs: {"task_id": task_id, **kwargs})
    monkeypatch.setattr(mod, "_torch_cuda_usable", lambda: True)
    monkeypatch.setattr(mod, "_nvidia_visible_devices_disabled", lambda: False)

    env = make_isaaclab_env(
        "isaaclab:Isaac-Cartpole-v0",
        headless=True,
        num_envs=1,
        device="cuda:0",
        render_mode="rgb_array",
        custom_option=3,
    )

    assert isinstance(env, IsaacLabGymEnvAdapter)
    task_id, kwargs = fake_gym.calls[-1]
    assert task_id == "Isaac-Cartpole-v0"
    assert kwargs["cfg"] == {
        "task_id": "Isaac-Cartpole-v0",
        "num_envs": 1,
        "device": "cuda:0",
    }
    assert kwargs["render_mode"] == "rgb_array"
    assert kwargs["custom_option"] == 3

    obs_space, action_space = resolve_isaaclab_env_spaces("isaaclab:Isaac-Cartpole-v0")

    assert obs_space.shape == (3,)
    assert action_space.shape == (2,)


def test_make_raw_isaaclab_env_returns_unadapted_env(monkeypatch):
    import problems.isaaclab_env_adapters as mod

    fake_gym = _FakeGym()
    monkeypatch.setattr(
        mod,
        "get_isaaclab_session",
        lambda **_kwargs: IsaacLabSession(app=None, gym=fake_gym),
    )
    monkeypatch.setattr(mod, "_torch_cuda_usable", lambda: True)
    monkeypatch.setattr(mod, "_nvidia_visible_devices_disabled", lambda: False)
    monkeypatch.setattr(
        mod,
        "_parse_env_cfg",
        lambda task_id, **kwargs: SimpleNamespace(task_id=task_id, seed=None, **kwargs),
    )

    env = make_raw_isaaclab_env(
        "isaaclab:Isaac-Cartpole-v0",
        num_envs=4,
        device="cuda:0",
        seed=11,
        batched=True,
    )

    task_id, kwargs = fake_gym.calls[-1]
    assert isinstance(env, _FakeIsaacEnv)
    assert task_id == "Isaac-Cartpole-v0"
    assert kwargs["cfg"].num_envs == 4
    assert kwargs["cfg"].device == "cuda:0"
    assert kwargs["cfg"].seed == 11
    assert "batched" not in kwargs


def test_disable_command_debug_visualizers():
    term = SimpleNamespace(debug_vis=True)
    cfg = SimpleNamespace(commands=SimpleNamespace(base_velocity=term))

    disable_command_debug_visualizers(cfg)

    assert term.debug_vis is False


def test_make_isaaclab_env_suppresses_setup_output(monkeypatch, capfd):
    import problems.isaaclab_env_adapters as mod

    fake_gym = _FakeGym()
    monkeypatch.setattr(
        mod,
        "get_isaaclab_session",
        lambda **_kwargs: IsaacLabSession(app=None, gym=fake_gym),
    )
    monkeypatch.setattr(mod, "_parse_env_cfg", lambda task_id, **kwargs: {"task_id": task_id, **kwargs})

    env = make_isaaclab_env("isaaclab:Isaac-Cartpole-v0", headless=True, num_envs=1)
    env.close()

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "Environment device" not in output
    assert "Number of environments" not in output
    assert "native setup detail" not in output


def test_capture_isaaclab_setup_output_replays_on_failure(capfd):
    import problems.isaaclab_env_adapters as mod

    try:
        with mod._capture_isaaclab_setup_output():
            print("setup detail")
            print("stderr setup detail", file=sys.stderr)
            os.write(2, b"native setup detail\n")
            raise RuntimeError("setup failed")
    except RuntimeError:
        pass

    captured = capfd.readouterr()
    assert captured.out == ""
    assert "setup detail" in captured.err
    assert "stderr setup detail" in captured.err
    assert "native setup detail" in captured.err


def test_isaaclab_task_list_main(monkeypatch, capsys):
    import problems.isaaclab_env_adapters as mod

    monkeypatch.setattr(
        mod,
        "list_isaaclab_tasks",
        lambda **kwargs: [f"task:{kwargs['keyword']}:{kwargs['headless']}"],
    )

    assert main(["--keyword", "cart", "--no-headless"]) == 0
    assert capsys.readouterr().out == "task:cart:False\n"
