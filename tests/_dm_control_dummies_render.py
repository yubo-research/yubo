import numpy as np
from _dm_control_dummies_spec import _DummyGlobal


class _DummyVis:
    global_ = _DummyGlobal()


class _DummyModel:
    ncam = 4
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
