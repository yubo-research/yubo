from scipy.stats import invgamma


class ScaledInverseChi2:
    def __init__(self, n, s2):
        self._alpha = n / 2
        self._beta = n * s2 / 2
        self._ig = invgamma(self._alpha)

    def rvs(self, size):
        x = self._ig.rvs(size=size)
        return self._beta * x

    def pdf(self, x):
        return self._ig.pdf(x / self._beta)
