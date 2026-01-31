import numpy as np
import torch


def test_torch_log_uniform_range():
    from sampling.log_uniform import torch_log_uniform

    torch.manual_seed(42)
    samples = [torch_log_uniform(0.01, 10.0) for _ in range(100)]
    for s in samples:
        assert 0.01 <= s <= 10.0


def test_np_log_uniform_range():
    from sampling.log_uniform import np_log_uniform

    np.random.seed(42)
    samples = [np_log_uniform(0.01, 10.0) for _ in range(100)]
    for s in samples:
        assert 0.01 <= s <= 10.0


def test_np_log_uniform_multiple():
    from sampling.log_uniform import np_log_uniform

    np.random.seed(42)
    samples = np_log_uniform(0.01, 10.0, num_samples=50)
    assert len(samples) == 50
    assert np.all(samples >= 0.01)
    assert np.all(samples <= 10.0)


def test_np_log_uniform_single():
    from sampling.log_uniform import np_log_uniform

    np.random.seed(42)
    s = np_log_uniform(1.0, 100.0, num_samples=1)
    assert isinstance(s, (float, np.floating))
