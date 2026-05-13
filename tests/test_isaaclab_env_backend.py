import sys
from types import SimpleNamespace

import numpy as np

from problems.isaaclab_env_adapters import (
    IsaacLabGymEnvAdapter,
    IsaacLabSession,
    get_isaaclab_session,
    is_isaaclab_env_tag,
    list_isaaclab_tasks,
    main,
    make_isaaclab_env,
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


def test_make_isaaclab_env_and_resolve_isaaclab_env_spaces(monkeypatch):
    import problems.isaaclab_env_adapters as mod

    fake_gym = _FakeGym()
    monkeypatch.setattr(
        mod,
        "get_isaaclab_session",
        lambda **_kwargs: IsaacLabSession(app=None, gym=fake_gym),
    )
    monkeypatch.setattr(mod, "_parse_env_cfg", lambda task_id, **kwargs: {"task_id": task_id, **kwargs})

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


def test_isaaclab_task_list_main(monkeypatch, capsys):
    import problems.isaaclab_env_adapters as mod

    monkeypatch.setattr(
        mod,
        "list_isaaclab_tasks",
        lambda **kwargs: [f"task:{kwargs['keyword']}:{kwargs['headless']}"],
    )

    assert main(["--keyword", "cart", "--no-headless"]) == 0
    assert capsys.readouterr().out == "task:cart:False\n"
