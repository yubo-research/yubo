import numpy as np


def make_boxed_gym_env_for_pixel_wrap(box):
    def reset(self, seed=None, options=None):
        _ = (seed, options)
        return np.zeros((2,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        return np.zeros((2,), dtype=np.float32), 0.0, False, False, {}

    def render(self):
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def close(self):
        return None

    return type(
        "_E",
        (),
        {
            "action_space": box,
            "metadata": {},
            "render_mode": "rgb_array",
            "observation_space": box,
            "__init__": lambda self: None,
            "reset": reset,
            "step": step,
            "render": render,
            "close": close,
        },
    )()
