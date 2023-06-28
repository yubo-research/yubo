import numpy as np


def test_pxmax():
    from sampling.pxmax import PXMax

    mu = np.array([0.5, 0.5])
    cov = np.array([1, 0.5])

    pxmax = PXMax(mu, cov, sigma_0=0.1)

    for s in pxmax.ask(10):
        print(s)
