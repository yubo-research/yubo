import numpy as np
from _dm_control_dummies_render import _DummyPhysics, _DummyVis
from _dm_control_dummies_spec import _DummySpec, _DummyTimeStep


class _DummyModelNoNames:
    ncam = 4
    vis = _DummyVis()
    cam_pos = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    cam_mode = np.asarray([2, 2, 0, 2], dtype=np.int32)
    cam_fovy = np.asarray([60.0, 45.0, 45.0, 50.0], dtype=np.float32)

    @staticmethod
    def name2id(name, kind):
        _ = name
        assert kind == "camera"
        return -1


class _DummyPhysicsNoNames:
    def __init__(self):
        self.model = _DummyModelNoNames()
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
    def _obs(position, velocity):
        return {
            "position": np.array(position, dtype=np.float32),
            "velocity": np.array(velocity, dtype=np.float32),
        }

    @staticmethod
    def reset():
        obs = _DummyDMEnv._obs([0.1, 0.2], [0.3])
        return _DummyTimeStep(observation=obs, reward=0.0, discount=1.0, is_last=False)

    @staticmethod
    def step(action):
        _ = action
        obs = _DummyDMEnv._obs([0.4, 0.5], [0.6])
        return _DummyTimeStep(observation=obs, reward=1.5, discount=0.95, is_last=True)

    @staticmethod
    def close():
        return None
