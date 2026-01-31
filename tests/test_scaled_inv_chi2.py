import numpy as np


def test_scaled_inv_chi2_rvs():
    from sampling.scaled_inv_chi2 import ScaledInverseChi2

    np.random.seed(42)
    rv = ScaledInverseChi2(n=5, s2=1.0)
    samples = rv.rvs(size=(100,))
    assert samples.shape == (100,)
    assert np.all(samples > 0)


def test_scaled_inv_chi2_pdf():
    from sampling.scaled_inv_chi2 import ScaledInverseChi2

    rv = ScaledInverseChi2(n=5, s2=1.0)
    pdf_val = rv.pdf(1.0)
    assert np.isfinite(pdf_val)
    assert pdf_val > 0
