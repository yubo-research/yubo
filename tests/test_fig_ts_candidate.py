def test_prob_inner():
    from figures.pts.fig_ts_candidate import prob_inner

    result = prob_inner(p_inner=0.5, num_samples=100, num_tries=100)
    assert 0.0 <= result <= 1.0
