import numpy as np


def test_arms_from_pareto_fronts():
    from types import SimpleNamespace

    from turbo_m_ref.turbo_1 import arms_from_pareto_fronts

    x_cand = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.3, 0.3]])
    mvn = SimpleNamespace(
        mu=np.array([1.0, 2.0, 3.0, 1.5]),
        se=np.array([0.1, 0.2, 0.3, 0.15]),
    )
    x_arms = arms_from_pareto_fronts(x_cand, mvn, num_arms=2)
    assert x_arms.shape == (2, 2)


def test_turbo_1_ask_tell_predict():
    from turbo_m_ref.turbo_1_ask_tell import Turbo1

    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    t = Turbo1(x_bounds=x_bounds, n_init=2, batch_size=1)
    t._X_best_so_far = np.array([0.5, 0.5])
    t._X_last = np.array([0.3, 0.3])

    x_best = t.predict(predict_best=True)
    assert np.allclose(x_best, [0.5, 0.5])

    x_last = t.predict(predict_best=False)
    assert np.allclose(x_last, [0.3, 0.3])


def test_turbo_1_ask_tell_maximize():
    from turbo_m_ref.turbo_1_ask_tell import Turbo1

    np.random.seed(42)

    def f(x):
        return -np.sum((x - 0.5) ** 2)

    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    t = Turbo1(x_bounds=x_bounds, n_init=3, batch_size=1)
    t.maximize(f, max_evals=5)
    assert t.n_evals >= 5


def test_turbo_1_randint_full_range():
    num_dim = 10
    dim_selected = set()
    for seed in range(5000):
        np.random.seed(seed)
        idx = np.random.randint(0, num_dim, size=1)[0]
        dim_selected.add(idx)

    assert len(dim_selected) == num_dim, (
        f"randint(0, num_dim) should select all dims; got {sorted(dim_selected)}"
    )
