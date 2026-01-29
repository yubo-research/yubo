class LassoBenchEnv:
    def __init__(self, pick_data, seed):
        from LassoBench import RealBenchmark

        import common.all_bounds as all_bounds

        self._bench = RealBenchmark(pick_data=pick_data, seed=seed)
        self.n_features = self._bench.n_features
        self.observation_space = all_bounds.get_box_1d01()
        self.action_space = all_bounds.get_box_bounds_x(self.n_features)

    def step(self, action):
        import numpy as np

        action = np.asarray(action, dtype=float)
        assert action.shape == (self.n_features,)
        assert np.all(action >= -1) and np.all(action <= 1)
        loss = self._bench.evaluate(action)
        return 1, -float(loss), True, None

    def reset(self, seed):
        return 0, None

    def close(self):
        pass
