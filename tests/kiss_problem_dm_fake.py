import numpy as np
from kiss_problem_dm_physics import DmPhysics
from kiss_problem_dm_ts import _TS, _Spec


class _FakeDM:
    physics = DmPhysics()

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
