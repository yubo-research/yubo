from _dm_control_dummies_env import _DummyDMEnv, _DummyPhysicsNoNames


class _DummyDMEnvNoNames(_DummyDMEnv):
    def __init__(self):
        self.physics = _DummyPhysicsNoNames()
