import numpy as np

from problems import dm_control_env
from problems.dm_control_env import DMControlEnv, _configure_headless_render_backend, _parse_env_name, make


class _DummySpec:
    def __init__(self, *, shape, dtype=np.float32, minimum=-1.0, maximum=1.0):
        self.shape = shape
        self.dtype = dtype
        self.minimum = np.full(shape, minimum, dtype=np.float32)
        self.maximum = np.full(shape, maximum, dtype=np.float32)


class _DummyTimeStep:
    def __init__(self, observation, reward, discount, is_last):
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._is_last = is_last

    def last(self):
        return self._is_last


class _DummyGlobal:
    offwidth = 640
    offheight = 480


class _DummyVis:
    global_ = _DummyGlobal()


class _DummyModel:
    vis = _DummyVis()

    @staticmethod
    def name2id(name, kind):
        assert kind == "camera"
        return 1 if name == "side" else -1


class _DummyPhysics:
    def __init__(self):
        self.model = _DummyModel()
        self.render_calls = []

    def render(self, **kwargs):
        self.render_calls.append(kwargs)
        return np.zeros((kwargs["height"], kwargs["width"], 3), dtype=np.uint8)


class _DummyDMEnv:
    def __init__(self):
        self.physics = _DummyPhysics()

    @staticmethod
    def observation_spec():
        return {"position": _DummySpec(shape=(2,)), "velocity": _DummySpec(shape=(1,))}

    @staticmethod
    def action_spec():
        return _DummySpec(shape=(3,), minimum=-2.0, maximum=2.0)

    @staticmethod
    def reset():
        obs = {
            "position": np.array([0.1, 0.2], dtype=np.float32),
            "velocity": np.array([0.3], dtype=np.float32),
        }
        return _DummyTimeStep(observation=obs, reward=0.0, discount=1.0, is_last=False)

    @staticmethod
    def step(action):
        _ = action
        obs = {
            "position": np.array([0.4, 0.5], dtype=np.float32),
            "velocity": np.array([0.6], dtype=np.float32),
        }
        return _DummyTimeStep(observation=obs, reward=1.5, discount=0.95, is_last=True)

    @staticmethod
    def close():
        return None


def test_parse_env_name():
    domain, task = _parse_env_name("dm_control/cheetah-run-v0")
    assert domain == "cheetah"
    assert task == "run"


def test_make_from_name(monkeypatch):
    monkeypatch.setattr(DMControlEnv, "_load_env", lambda self, seed: _DummyDMEnv())
    env = make("dm_control/cheetah-run-v0")
    assert isinstance(env, DMControlEnv)


def test_dm_control_env_step_and_render(monkeypatch):
    monkeypatch.setattr(DMControlEnv, "_load_env", lambda self, seed: _DummyDMEnv())
    env = DMControlEnv("cheetah", "run", render_mode="rgb_array")

    obs, info = env.reset()
    assert info == {}
    assert obs.shape == (3,)

    action = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    next_obs, reward, terminated, truncated, extra = env.step(action)
    assert next_obs.shape == (3,)
    assert reward == 1.5
    assert terminated is True
    assert truncated is False
    assert extra == {"discount": 0.95}

    frame = env.render()
    assert frame.shape == (720, 1280, 3)

    render_call = env._env.physics.render_calls[-1]
    assert render_call["width"] == 1280
    assert render_call["height"] == 720


def test_configure_headless_render_backend_linux_no_display(monkeypatch):
    monkeypatch.setattr(dm_control_env.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    _configure_headless_render_backend("rgb_array")

    assert dm_control_env.os.environ["MUJOCO_GL"] == "egl"
    assert dm_control_env.os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_configure_headless_render_backend_respects_existing_setting(monkeypatch):
    monkeypatch.setattr(dm_control_env.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    _configure_headless_render_backend("rgb_array")

    assert dm_control_env.os.environ["MUJOCO_GL"] == "glfw"
    assert "PYOPENGL_PLATFORM" not in dm_control_env.os.environ
