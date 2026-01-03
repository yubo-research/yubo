import math

from uhd.trust_region import TrustRegionAdjustor, TrustRegionConfig


def test_trust_region_default_config_and_reset():
    tr = TrustRegionAdjustor(dim=10, batch_size=2)
    assert 0.0 < tr.length <= tr.config.length_max
    init_len = tr.length
    tr.update([1.0, 1.1])
    assert tr.length == init_len
    tr.reset()
    assert math.isclose(tr.length, tr.config.length_init)


def test_grow_on_successes_and_cap_by_max():
    tr = TrustRegionAdjustor(dim=8, batch_size=1, config=TrustRegionConfig(length_min=0.5**7, length_max=1.6, length_init=0.2, succtol=2, failtol=10))
    l0 = tr.length
    tr.update([1.0])
    tr.update([1.01])
    tr.update([1.02])
    assert math.isclose(tr.length, min(2.0 * l0, tr.config.length_max))
    l1 = tr.length
    tr.update([1.03])
    tr.update([1.04])
    assert math.isclose(tr.length, min(2.0 * l1, tr.config.length_max))


def test_shrink_on_failures_in_steps_of_two():
    tr = TrustRegionAdjustor(dim=20, batch_size=2, config=TrustRegionConfig(length_min=0.5**7, length_max=1.6, length_init=0.8, succtol=3, failtol=2))
    initial_length = tr.length
    tr.update([1.0, 1.1])
    tr.update([1.0, 1.1])
    tr.update([1.0, 1.1])
    assert math.isclose(tr.length, initial_length / 2.0)
    tr.update([1.0, 1.1])
    tr.update([1.0, 1.1])
    assert math.isclose(tr.length, initial_length / 4.0)


def test_relative_improvement_threshold():
    tr = TrustRegionAdjustor(dim=5, batch_size=1, config=TrustRegionConfig(length_min=0.5**7, length_max=1.6, length_init=0.5, succtol=2, failtol=3))
    tr.update([100.0])
    initial_length = tr.length
    tr.update([100.09])
    assert math.isclose(tr.length, initial_length)
    tr.update([100.21])
    tr.update([100.32])
    assert math.isclose(tr.length, min(2.0 * initial_length, tr.config.length_max))


def test_failtol_default_matches_reference_formula():
    dim = 40
    bs = 5
    tr = TrustRegionAdjustor(dim=dim, batch_size=bs)
    expected = int(math.ceil(max(4.0 / bs, dim / bs)))
    assert tr.config.failtol == expected


def test_length_never_below_min_on_many_failures():
    tr = TrustRegionAdjustor(
        dim=10,
        batch_size=1,
        config=TrustRegionConfig(length_min=1e-3, length_max=1.6, length_init=0.8, succtol=10, failtol=1),
    )
    for _ in range(2000):
        tr.update([0.0])
    assert tr.length >= tr.config.length_min
    assert math.isclose(tr.length, tr.config.length_min)


def test_length_capped_at_max_on_many_successes():
    tr = TrustRegionAdjustor(
        dim=10,
        batch_size=1,
        config=TrustRegionConfig(length_min=0.5**7, length_max=0.9, length_init=0.1, succtol=1, failtol=1000),
    )
    base = 1.0
    for _ in range(100):
        base += 1.0
        tr.update([base])
        if tr.length >= tr.config.length_max:
            break
    assert tr.length <= tr.config.length_max
    assert math.isclose(tr.length, tr.config.length_max)
