import numpy as np


def test_normalizer_new_init():
    from problems.normalizer_new import NormalizerNew

    norm = NormalizerNew(shape=(3,), num_init=1)
    assert norm._num == 1


def test_normalizer_new_init_zero():
    from problems.normalizer_new import NormalizerNew

    norm = NormalizerNew(shape=(3,), num_init=0)
    assert norm._num == 0


def test_normalizer_new_update():
    from problems.normalizer_new import NormalizerNew

    norm = NormalizerNew(shape=(3,), num_init=0)
    norm.update(np.array([1.0, 2.0, 3.0]))
    assert norm._num == 1


def test_normalizer_new_mean_and_std():
    from problems.normalizer_new import NormalizerNew

    norm = NormalizerNew(shape=(3,), num_init=0)
    norm.update(np.array([1.0, 2.0, 3.0]))
    norm.update(np.array([1.0, 2.0, 3.0]))
    mean, std = norm.mean_and_std()
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.allclose(mean, [1.0, 2.0, 3.0])
