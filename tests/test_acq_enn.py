import numpy as np
import pytest

from acq.acq_enn import AcqENN, ENNConfig


def setup_acq_enn(num_dim=2, k=3, k_novelty=None):
    config = ENNConfig(k=k, k_novelty=k_novelty)
    acq = AcqENN(num_dim, config)
    return acq, config


def setup_with_data(acq, num_points=10):
    x_train = np.random.uniform(size=(num_points, acq._num_dim))
    y_train = np.random.normal(size=(num_points, 1))
    d_train = np.random.normal(size=(num_points, 2)) if acq._config.k_novelty is not None else None
    acq.add(x_train, y_train, d_train)
    return x_train, y_train, d_train


def test_enn_config():
    config = ENNConfig(k=5)
    assert config.k == 5
    assert config.num_candidates_per_arm == 0
    assert config.stagger is False
    assert config.acq == "pareto"
    assert config.num_over_sample_per_arm == 1
    assert config.candidate_generator == "sobol"
    assert config.tr_type == "mean"
    assert config.raasp_type is None
    assert config.k_novelty is None


def test_enn_config_post_init():
    with pytest.raises(AssertionError):
        ENNConfig(k=0, num_over_sample_per_arm=0)


def test_acq_enn_initialization():
    acq, config = setup_acq_enn()
    assert acq._num_dim == 2
    assert acq._config == config
    assert acq._enn is None
    assert acq._enn_d is None
    assert len(acq._x_train) == 0
    assert len(acq._y_train) == 0
    assert acq._d_train is None


def test_add_empty_data():
    acq, _ = setup_acq_enn()
    acq.add([], [])
    assert len(acq._x_train) == 0
    assert len(acq._y_train) == 0


def test_add_first_data():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)

    assert acq._enn is not None
    assert len(acq._x_train) == 5
    assert len(acq._y_train) == 5
    assert np.array_equal(acq._x_train, x_train)
    assert np.array_equal(acq._y_train, y_train)


def test_add_with_descriptors():
    acq, _ = setup_acq_enn(k_novelty=3)
    x_train, y_train, d_train = setup_with_data(acq, 5)

    assert acq._enn_d is not None
    assert acq._d_train is not None
    assert len(acq._d_train) == 5
    assert np.array_equal(acq._d_train, d_train)


def test_add_incremental():
    acq, _ = setup_acq_enn()
    x1, y1, _ = setup_with_data(acq, 3)

    x2 = np.random.uniform(size=(2, acq._num_dim))
    y2 = np.random.normal(size=(2, 1))
    acq.add(x2, y2)

    assert len(acq._x_train) == 5
    assert len(acq._y_train) == 5


def test_keep_top_n():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    x_kept, y_kept = acq.keep_top_n(5)
    assert len(x_kept) == 5
    assert len(y_kept) == 5
    assert len(acq._x_train) == 5
    assert len(acq._y_train) == 5


def test_keep_top_n_no_reduction():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_kept, y_kept = acq.keep_top_n(10)
    assert len(x_kept) == 5
    assert len(y_kept) == 5


def test_sample_segments():
    acq, _ = setup_acq_enn()
    x_0 = np.random.uniform(size=(3, acq._num_dim))
    x_far = np.random.uniform(size=(3, acq._num_dim))

    x_cand = acq._sample_segments(x_0, x_far)
    assert x_cand.shape == (3, acq._num_dim)

    for i in range(3):
        assert np.all(x_cand[i] >= np.minimum(x_0[i], x_far[i]))
        assert np.all(x_cand[i] <= np.maximum(x_0[i], x_far[i]))


def test_sample_segments_stagger():
    acq, config = setup_acq_enn()
    config.stagger = True
    x_0 = np.random.uniform(size=(3, acq._num_dim))
    x_far = np.random.uniform(size=(3, acq._num_dim))

    x_cand = acq._sample_segments(x_0, x_far)
    assert x_cand.shape == (3, acq._num_dim)


def test_select_pivots():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    pivots = acq._select_pivots(3)
    assert pivots.shape == (3, acq._num_dim)


def test_i_pareto_fronts_discrepancy():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    i_keep = acq._i_pareto_fronts_discrepancy(5)
    assert len(i_keep) == 5
    assert all(0 <= i < len(x_train) for i in i_keep)


def test_i_pareto_fronts_strict():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    x_cand = np.random.uniform(size=(8, acq._num_dim))
    i_keep = acq._i_pareto_fronts_strict(x_cand, 3)
    assert len(i_keep) == 3
    assert all(0 <= i < len(x_cand) for i in i_keep)


def test_i_pareto_fronts_strict_exclude_nearest():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    x_cand = np.random.uniform(size=(8, acq._num_dim))
    i_keep = acq._i_pareto_fronts_strict(x_cand, 3)
    assert len(i_keep) == 3


def test_xxx_i_pareto_fronts_strict():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    x_cand = np.random.uniform(size=(8, acq._num_dim))
    i_keep = acq._i_pareto_fronts_strict(x_cand, 3)
    assert len(i_keep) == 3


def test_pareto_fronts_strict():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 10)

    x_cand = np.random.uniform(size=(8, acq._num_dim))
    x_arms = acq._pareto_fronts_strict(x_cand, 3)
    assert x_arms.shape == (3, acq._num_dim)


def test_uniform():
    acq, _ = setup_acq_enn()
    x_cand = np.random.uniform(size=(10, acq._num_dim))

    x_arms = acq._uniform(x_cand, 3)
    assert x_arms.shape == (3, acq._num_dim)


def test_edn():
    acq, _ = setup_acq_enn(k_novelty=3)
    x_train, y_train, d_train = setup_with_data(acq, 10)

    x_cand = np.random.uniform(size=(8, acq._num_dim))
    mu_se = np.random.uniform(size=(8, 2))

    dns, dns_se = acq._edn(x_cand, mu_se)
    assert dns.shape == (8, 1)
    assert dns_se.shape == (8, 1)


def test_dominated_novelty_selection():
    acq, _ = setup_acq_enn(k_novelty=3)
    x_train, y_train, d_train = setup_with_data(acq, 10)

    x_cand = np.random.uniform(size=(8, acq._num_dim))
    x_arms = acq._dominated_novelty_selection(x_cand, 3)
    assert x_arms.shape == (3, acq._num_dim)


def test_dominated_novelty_selection_empty():
    acq, _ = setup_acq_enn(k_novelty=3)
    x_cand = np.random.uniform(size=(8, acq._num_dim))
    x_arms = acq._dominated_novelty_selection(x_cand, 3)
    assert x_arms.shape == (3, acq._num_dim)


def test_draw_sobol():
    acq, _ = setup_acq_enn()
    x_cand = acq._draw_sobol(10)
    assert x_cand.shape == (10, acq._num_dim)
    assert np.all(x_cand >= 0)
    assert np.all(x_cand <= 1)


def test_draw_sobol_with_bounds():
    acq, _ = setup_acq_enn()
    bounds = np.array([[0.2, 0.8], [0.3, 0.7]])
    x_cand = acq._draw_sobol(10, bounds)
    assert x_cand.shape == (10, acq._num_dim)
    assert np.all(x_cand >= 0.0)
    assert np.all(x_cand <= 1.0)


def test_raasp():
    acq, config = setup_acq_enn()
    config.raasp_type = "raasp"
    x_center = np.random.uniform(size=(1, acq._num_dim))

    x_cand = acq._raasp(x_center, 10)
    assert x_cand is not None
    assert x_cand.shape == (10, acq._num_dim)


def test_raasp_none():
    acq, config = setup_acq_enn()
    config.raasp_type = "invalid"
    x_center = np.random.uniform(size=(1, acq._num_dim))

    x_cand = acq._raasp(x_center, 10)
    assert x_cand is None


def test_raasp_or_sobol():
    acq, config = setup_acq_enn()
    config.raasp_type = "raasp"
    x_center = np.random.uniform(size=(1, acq._num_dim))

    x_cand = acq._raasp_or_sobol(x_center, 10)
    assert x_cand.shape == (10, acq._num_dim)


def test_raasp_or_sobol_fallback():
    acq, config = setup_acq_enn()
    config.raasp_type = "invalid"
    x_center = np.random.uniform(size=(1, acq._num_dim))

    x_cand = acq._raasp_or_sobol(x_center, 10)
    assert x_cand.shape == (10, acq._num_dim)


def test_trust_region_insufficient_data():
    acq, _ = setup_acq_enn()
    x_cand = acq._trust_region(10)
    assert x_cand.shape == (10, acq._num_dim)


def test_trust_region():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._trust_region(10)
    assert x_cand.shape == (10, acq._num_dim)


def test_candidates_sobol():
    acq, config = setup_acq_enn()
    config.candidate_generator = "sobol"
    config.num_candidates_per_arm = 2

    x_cand = acq._candidates(3)
    assert x_cand.shape == (6, acq._num_dim)


def test_candidates_tr():
    acq, config = setup_acq_enn()
    config.candidate_generator = "tr"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._candidates(3)
    assert x_cand.shape == (6, acq._num_dim)


def test_candidates_best():
    acq, config = setup_acq_enn()
    config.candidate_generator = "best"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._candidates(3)
    assert x_cand.shape == (6, acq._num_dim)


def test_candidates_pivots():
    acq, config = setup_acq_enn()
    config.candidate_generator = "pivots"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._candidates(3)
    assert x_cand.shape == (6, acq._num_dim)


def test_candidates_rand_train():
    acq, config = setup_acq_enn()
    config.candidate_generator = "rand_train"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._candidates(3)
    assert x_cand.shape == (6, acq._num_dim)


def test_candidates_convex():
    acq, config = setup_acq_enn()
    config.candidate_generator = "convex"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._candidates(3)
    assert x_cand.shape[1] == acq._num_dim


def test_candidates_invalid():
    acq, config = setup_acq_enn()
    config.candidate_generator = "invalid"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    with pytest.raises(AssertionError):
        acq._candidates(3)


def test_candidates_multiple():
    acq, config = setup_acq_enn()
    config.candidate_generator = "sobol+tr"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_cand = acq._candidates(3)
    assert x_cand.shape[1] == acq._num_dim


def test_draw_two_level():
    acq, config = setup_acq_enn()
    config.acq = "pareto_strict"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_arms = acq._draw_two_level(3)
    assert x_arms.shape == (3, acq._num_dim)


def test_draw_two_level_dominated_novelty():
    acq, config = setup_acq_enn()
    config.acq = "dominated_novelty"
    config.k_novelty = 3
    config.num_candidates_per_arm = 2
    x_train, y_train, d_train = setup_with_data(acq, 5)

    x_arms = acq._draw_two_level(3)
    assert x_arms.shape == (3, acq._num_dim)


def test_draw():
    acq, config = setup_acq_enn()
    config.acq = "pareto_strict"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_arms = acq.draw(3)
    assert x_arms.shape == (3, acq._num_dim)


def test_draw_uniform():
    acq, config = setup_acq_enn()
    config.acq = "uniform"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    x_arms = acq.draw(3)
    assert x_arms.shape == (3, acq._num_dim)


def test_draw_invalid():
    acq, config = setup_acq_enn()
    config.acq = "invalid"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)

    with pytest.raises(AssertionError):
        acq.draw(3)


def test_biased_raasp():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)
    x_center = np.random.uniform(size=(1, acq._num_dim))

    x_cand = acq._biased_raasp(x_center, 10)
    assert x_cand.shape[1] == acq._num_dim
    assert len(x_cand) > 10


def test_trust_region_different_types():
    acq, config = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)

    for tr_type in ["mean", "0", "median"]:
        config.tr_type = tr_type
        x_cand = acq._trust_region(10)
        assert x_cand.shape == (10, acq._num_dim)


def test_trust_region_invalid_type():
    acq, config = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)
    config.tr_type = "invalid"

    with pytest.raises(AssertionError):
        acq._trust_region(10)


def test_sample_segments_empty():
    acq, _ = setup_acq_enn()
    x_0 = np.empty((0, acq._num_dim))
    x_far = np.empty((0, acq._num_dim))
    x_cand = acq._sample_segments(x_0, x_far)
    assert x_cand.shape == (0, acq._num_dim)


def test_uniform_empty():
    acq, _ = setup_acq_enn()
    x_cand = np.empty((0, acq._num_dim))
    with pytest.raises(ValueError):
        acq._uniform(x_cand, 1)


def test_uniform_more_than_available():
    acq, _ = setup_acq_enn()
    x_cand = np.random.uniform(size=(2, acq._num_dim))
    with pytest.raises(ValueError):
        acq._uniform(x_cand, 3)


def test_sample_segments_stagger_batch():
    acq, config = setup_acq_enn()
    config.stagger = True
    x_0 = np.random.uniform(size=(10, acq._num_dim))
    x_far = np.random.uniform(size=(10, acq._num_dim))
    x_cand = acq._sample_segments(x_0, x_far)
    assert x_cand.shape == (10, acq._num_dim)
    assert np.all(x_cand >= np.minimum(x_0, x_far))
    assert np.all(x_cand <= np.maximum(x_0, x_far))


def test_draw_two_level_invalid_acq():
    acq, config = setup_acq_enn()
    config.acq = "not_a_real_acq"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)
    with pytest.raises(AssertionError):
        acq._draw_two_level(3)


def test_draw_zero_arms():
    acq, config = setup_acq_enn()
    config.acq = "pareto_strict"
    config.num_candidates_per_arm = 2
    x_train, y_train, _ = setup_with_data(acq, 5)
    with pytest.raises(RuntimeError):
        acq.draw(0)


def test_keep_top_n_zero():
    acq, _ = setup_acq_enn()
    x_train, y_train, _ = setup_with_data(acq, 5)
    with pytest.raises(IndexError):
        acq.keep_top_n(0)
