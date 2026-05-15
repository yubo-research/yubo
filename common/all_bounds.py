import numpy as np
from gymnasium.spaces import Box

x_low = -1.0
x_high = 1.0
x_width = x_high - x_low

p_low = -1.0
p_high = 1.0
p_width = p_high - p_low

bt_low = 0.0
bt_high = 1.0
bt_width = bt_high - bt_low


def get_box_bounds_x(num_dim):
    ones = np.ones(num_dim, dtype=np.float32)
    return Box(
        low=x_low * ones,
        high=x_high * ones,
    )


def get_box_1d01():
    return Box(low=0.0, high=1.0, dtype=np.float32)
