def test_gelman_rubin():
    import numpy as np

    from sampling.gelman_rubin import GelmanRubin

    np.random.seed(17)

    gr = GelmanRubin()
    for _ in range(30):
        gr.append(np.random.uniform(size=(10, 3)))
    r_hat = gr.r_hat()
    print(r_hat)
    assert r_hat >= 1
    assert r_hat < 1.2
