from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def test_kiss_cov_problem_env_conf_backends_and_env_conf(monkeypatch):
    from problems.env_conf import needs_atari_dm_bindings, register_atari_dm_bindings_loader
    from problems.env_conf_backends import AtariDMBindings

    called = {"loader": 0}

    def _loader():
        called["loader"] += 1
        return SimpleNamespace()

    register_atari_dm_bindings_loader(_loader)
    assert needs_atari_dm_bindings("atari:Pong")
    assert needs_atari_dm_bindings("dm_control/cartpole-swingup-v0")
    assert not needs_atari_dm_bindings("f:sphere-2d")

    b = AtariDMBindings(
        resolve_dm_control_from_tag=lambda tag, use_pixels: ("dm_control/cartpole-swingup-v0", object),
        resolve_atari_from_tag=lambda tag: ("ALE/Pong-v5", object),
        make_atari_preprocess_options=lambda **kwargs: kwargs,
        make_dm_control_env=lambda *args, **kwargs: None,
        make_atari_env=lambda *args, **kwargs: None,
    )
    assert callable(b.resolve_atari_from_tag)


def test_kiss_cov_problem_atari_env(monkeypatch):
    class _ALE:
        def __init__(self):
            self._n = 0
            self._lives = 3

        def setInt(self, *_args):
            return None

        def setFloat(self, *_args):
            return None

        def setBool(self, *_args):
            return None

        def loadROM(self, *_args):
            return None

        def getMinimalActionSet(self):
            return [0, 1]

        def getLegalActionSet(self):
            return [0, 1]

        def getScreenGrayscale(self):
            return np.zeros((84, 84), dtype=np.uint8)

        def getScreenRGB(self):
            return np.zeros((84, 84, 3), dtype=np.uint8)

        def act(self, _a):
            self._n += 1
            return 1.0

        def game_over(self):
            return self._n > 3

        def reset_game(self):
            self._n = 0

        def lives(self):
            return self._lives

    fake_ale_py = SimpleNamespace(
        ALEInterface=_ALE,
        roms=SimpleNamespace(get_rom_path=lambda _rom_id: "/tmp/fake"),
    )
    monkeypatch.setitem(__import__("sys").modules, "ale_py", fake_ale_py)

    import problems.atari_env as atari_env

    env = atari_env.ALEAtariEnv("ALE/Pong-v5", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 4
    obs2, reward, terminated, truncated, _ = env.step(0)
    assert obs2.shape[0] == 4
    assert isinstance(reward, float)
    _ = env.render()
    env.close()
    wrapped = atari_env.make("atari:Pong")
    assert wrapped is not None


def test_kiss_cov_problem_dm_control_and_pixel_policies(monkeypatch):
    import problems.dm_control_env as dm_env
    from problems.pixel_policies import AtariAgent57LitePolicy, AtariCNNPolicy, AtariGaussianPolicy

    class _TS:
        def __init__(self, last=False):
            self.observation = {"state": np.zeros((4,), dtype=np.float32)}
            self.reward = 1.0
            self._last = last
            self.discount = 1.0

        def last(self):
            return self._last

    class _Spec:
        def __init__(self, shape, minimum=None, maximum=None, dtype=np.float32):
            self.shape = shape
            self.minimum = minimum
            self.maximum = maximum
            self.dtype = dtype

    class _Physics:
        class _Model:
            vis = SimpleNamespace(global_=SimpleNamespace(offwidth=1280, offheight=720))
            ncam = 1
            cam_pos = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            cam_mode = np.array([1], dtype=np.int32)
            cam_fovy = np.array([45.0], dtype=np.float32)

            @staticmethod
            def name2id(_name, _kind):
                return 0

        model = _Model()

        @staticmethod
        def render(width, height, camera_id):
            return np.zeros((height, width, 3), dtype=np.uint8)

    class _FakeDM:
        physics = _Physics()

        @staticmethod
        def observation_spec():
            return {"state": _Spec((4,), minimum=-np.ones(4), maximum=np.ones(4))}

        @staticmethod
        def action_spec():
            return _Spec((2,), minimum=-np.ones(2), maximum=np.ones(2))

        @staticmethod
        def reset():
            return _TS(last=False)

        def step(self, action):
            _ = action
            return _TS(last=False)

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(dm_env.suite, "load", lambda *args, **kwargs: _FakeDM())
    env = dm_env.make("dm_control/cartpole-swingup-v0")
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    _ = env.step(np.zeros((2,), dtype=np.float32))
    _ = env.render()
    env.close()

    pix = dm_env.make("dm_control/cartpole-swingup-v0", from_pixels=True, pixels_only=True)
    pobs, _ = pix.reset(seed=0)
    assert pobs.shape[-1] == 3
    _ = pix.step(np.zeros((2,), dtype=np.float32))
    pix.close()

    env_conf = SimpleNamespace(
        problem_seed=0,
        state_space=SimpleNamespace(shape=(4, 84, 84)),
        action_space=SimpleNamespace(n=6),
    )
    cnn = AtariCNNPolicy(env_conf, hidden_sizes=(16,), variant="small")
    a1 = cnn(np.zeros((4, 84, 84), dtype=np.float32))
    assert isinstance(a1, int)

    gauss = AtariGaussianPolicy(env_conf, hidden_sizes=(16,), variant="small")
    a2 = gauss(np.zeros((4, 84, 84), dtype=np.float32))
    assert isinstance(a2, int)

    agent57 = AtariAgent57LitePolicy(env_conf, lstm_hidden=16, cnn_variant="small")
    agent57._h = torch.zeros((1, 1, agent57._lstm_hidden), dtype=torch.float32)
    agent57._c = torch.zeros((1, 1, agent57._lstm_hidden), dtype=torch.float32)
    a3 = agent57(np.zeros((4, 84, 84), dtype=np.float32))
    assert isinstance(a3, int)
    agent57.reset_state()


def test_kiss_cov_problem_dm_control_direct_units():
    from problems.dm_control_env import BoxSpace, DictSpace, _PixelObsWrapper

    box = BoxSpace(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
    sample = box.sample()
    assert sample.shape == (2,)

    dct = DictSpace({"x": box})
    out = dct.sample()
    assert "x" in out

    class _E:
        action_space = box
        metadata = {}
        render_mode = "rgb_array"
        observation_space = box

        @staticmethod
        def reset(seed=None, options=None):
            _ = (seed, options)
            return np.zeros((2,), dtype=np.float32), {}

        @staticmethod
        def step(action):
            _ = action
            return np.zeros((2,), dtype=np.float32), 0.0, False, False, {}

        @staticmethod
        def render():
            return np.zeros((84, 84, 3), dtype=np.uint8)

        @staticmethod
        def close():
            return None

    wrapped = _PixelObsWrapper(_E(), pixels_only=True, size=84)
    obs, _ = wrapped.reset(seed=0)
    assert obs.shape == (84, 84, 3)
    _ = wrapped.step(np.zeros((2,), dtype=np.float32))
    wrapped.close()
