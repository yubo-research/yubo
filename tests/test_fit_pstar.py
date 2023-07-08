def _test_pstar(nu, sigma, eps):
    import numpy as np

    from sampling.fit_pstar import FitPStar

    np.random.seed(17)
    mu = np.array([0.5, 0.5])
    cov = np.array([1, 0.1])

    pstar = FitPStar(mu, cov)

    def _gp(x, nu):
        return -((x - 0.5) ** 2).sum() + nu * np.random.normal()

    def thompson_sample(x, p, nu):
        s_max = None
        y_max = -1e99
        for xx, pp in zip(x, p):
            y = _gp(xx, nu)
            if y > y_max:
                y_max = y
                s_max = (xx, pp)
        return s_max

    num = 30
    while not pstar.converged():
        x, p = pstar.ask(num)
        s = x.std(axis=0)

        resamples = []
        for _ in range(len(x)):
            resamples.append(thompson_sample(x, p, nu))
        print("S:", pstar.sigma())
        pstar.tell(*zip(*resamples))

    assert s[0] > 2 * s[1], s
    assert (pstar.sigma() - sigma) < eps


def test_pstar_000():
    _test_pstar(nu=0, sigma=1e-5, eps=1e-4)


def test_pstar_001():
    _test_pstar(nu=0.001, sigma=0.01, eps=0.01)
