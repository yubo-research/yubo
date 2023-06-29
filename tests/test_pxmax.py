def _test_pxmax(nu, sigma, eps):
    import numpy as np

    from sampling.pxmax import PXMax

    np.random.seed(17)
    mu = np.array([0.5, 0.5])
    cov = np.array([1, 0.1])

    pxmax = PXMax(mu, cov, sigma_0=0.1)

    def _gp(x, nu):
        return -((x - 0.5) ** 2).sum() + nu * np.random.normal()

    def thompson_sample(samples, nu):
        s_max = None
        y_max = -1e99
        for s in samples:
            y = _gp(s.x, nu)
            if y > y_max:
                y_max = y
                s_max = s
        return s_max

    num = 30
    for _ in range(10):
        samples = pxmax.ask(num)
        s = np.vstack([s.x for s in samples]).std(axis=0)
        assert s[0] > 2 * s[1], s

        resamples = []
        for _ in samples:
            resamples.append(thompson_sample(samples, nu))
        pxmax.tell(resamples)
        # print (num, pxmax.sigma())
    assert (pxmax.sigma() - sigma) < eps


def test_pxmax_000():
    _test_pxmax(nu=0, sigma=0, eps=1e-6)


def test_pxmax_001():
    _test_pxmax(nu=0.001, sigma=0.01, eps=0.01)
