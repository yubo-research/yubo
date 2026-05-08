from types import SimpleNamespace

import numpy as np


class DmModel:
    vis = SimpleNamespace(global_=SimpleNamespace(offwidth=1280, offheight=720))
    ncam = 1
    cam_pos = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    cam_mode = np.array([1], dtype=np.int32)
    cam_fovy = np.array([45.0], dtype=np.float32)

    @staticmethod
    def name2id(_name, _kind):
        return 0


class DmPhysics:
    model = DmModel()

    @staticmethod
    def render(width, height, camera_id):
        return np.zeros((height, width, 3), dtype=np.uint8)
