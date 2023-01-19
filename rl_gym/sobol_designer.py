from scipy.stats import qmc


class SobolDesigner:
    def __init__(self, num_dim, max_points=100):
        self._xs = qmc.Sobol(num_dim).random(max_points)

    def get(self):
        x = self._xs[0, :]
        self._xs = self._xs[1:, :]
        return x
