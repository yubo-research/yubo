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
        i_max = None
        for i, (xx, pp) in enumerate(zip(x, p)):
            y = _gp(xx, nu)
            if y > y_max:
                y_max = y
                s_max = (xx, pp)
                i_max = i
        return i_max, s_max

    def calc_p_max(x, p, nu):
        p_max = np.zeros(shape=(len(x),))
        for _ in range(1024):
            i_max, _ = thompson_sample(x, p, nu)
            p_max[i_max] += 1
        p_max = p_max / p_max.sum()
        return p_max

    num = 30
    for _ in range(30):
        x, p = cem.ask(num)
        s = x.std(axis=0)

        if False:
            resamples = []
            for _ in range(len(x)):
                resamples.append(thompson_sample(x, p, nu))

        p_max = calc_p_max(x, p, nu)
        print("S:", cem.sigma())
        if cem.sigma() < 1e-9:
            break
        cem.tell(x, p, p_max)
        # *zip(*resamples),

    assert s[0] > 2 * s[1], s
    assert (cem.sigma() - sigma) < eps


def test_cem_000():
    _test_cem(nu=0, sigma=1e-9, eps=1e-8)


def test_cem_001():
    _test_cem(nu=0.001, sigma=0.01, eps=0.01)
