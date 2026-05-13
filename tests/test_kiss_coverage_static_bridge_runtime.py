from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_kiss_bridge_rl_core_runtime():
    from rl.core.runtime import mps_is_available, obs_scale_from_env, seed_everything

    seed_everything(0)
    _ = mps_is_available()

    Gym = type(
        "Gym",
        (),
        {
            "transform_state": True,
            "state_space": SimpleNamespace(
                low=np.zeros(3, dtype=np.float32),
                high=np.ones(3, dtype=np.float32),
                shape=(3,),
            ),
        },
    )
    E = type(
        "E",
        (),
        {
            "gym_conf": Gym(),
            "observation_space": SimpleNamespace(shape=(3,)),
            "ensure_spaces": lambda self: None,
        },
    )
    obs_scale_from_env(E())
