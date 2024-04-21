def test_parallel_mcmc_convergence():
    import numpy as np

    from sampling.parallel_mcmc_convergence import ParallelMCMCConvergence

    np.random.seed(17)

    pmc = ParallelMCMCConvergence()
    n = 128
    assert not pmc.converged(np.random.normal(size=(n, 3)))

    assert not pmc.converged(1 + np.random.normal(size=(n, 3)))
    assert not pmc.converged(3 * np.random.normal(size=(n, 3)))

    assert pmc.converged(np.random.normal(size=(n, 3)))
