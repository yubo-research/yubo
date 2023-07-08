def _test_cem(nu, sigma, eps):
    import numpy as np

    from sampling.cem_scale import CEMScale

    np.random.seed(17)
    mu = np.array([0.5, 0.5])
    cov = np.array([1, 0.1])

    cem = CEMScale(mu, cov, sigma_0=0.3)

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
    for _ in range(30):
        # while not cem.converged():
        x, p = cem.ask(num)
        s = x.std(axis=0)

        resamples = []
        for _ in range(len(x)):
            resamples.append(thompson_sample(x, p, nu))
        print("S:", cem.sigma())
        if cem.sigma() < 1e-9:
            break
        cem.tell(*zip(*resamples))

    assert s[0] > 2 * s[1], s
    assert (cem.sigma() - sigma) < eps


def test_cem_000():
    _test_cem(nu=0, sigma=1e-9, eps=1e-8)


def test_cem_001():
    _test_cem(nu=0.001, sigma=0.01, eps=0.01)
