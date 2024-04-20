def test_gelman_rubin():
    import numpy as np

    from sampling.gelman_rubin import GelmanRubin

    np.random.seed(17)

    gr = GelmanRubin()
    for _ in range(12):
        gr.append(np.random.uniform(size=(10,)))
    assert gr.get() < 1.2
