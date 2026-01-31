import numpy as np
import pytest


@pytest.mark.parametrize(
    "num_init,expected_num",
    [
        (1, 1),
        (0, 0),
    ],
)
def test_normalizer_new_init(num_init, expected_num):
    from problems.normalizer_new import NormalizerNew

    norm = NormalizerNew(shape=(3,), num_init=num_init)
    assert norm._num == expected_num


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
