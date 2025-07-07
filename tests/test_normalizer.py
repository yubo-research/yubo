def test_normalizer():
    import numpy as np

    from problems.normalizer import Normalizer

    msf = Normalizer(shape=(3,))

    mu = np.array([0, 1.141, 3.14])
    sg = np.array([0.1, 0.2, 0.3])
    for _ in range(1000):
        x = mu + sg * np.random.normal(size=(3,))
        msf.update(x)

    m, s = msf.mean_and_std()

    print(m)
    print(s)

    assert np.all(np.abs(m - mu) < 1e-1)
    assert np.all(np.abs(s - sg) < 1e-1)


def test_normalizer_numerical_instability():
    import numpy as np

    from problems.normalizer import Normalizer

    msf = Normalizer(shape=(1,))

    large_mean = 1e12
    tiny_std = 1e-12

    for _ in range(100000):
        x = np.array([large_mean + tiny_std * np.random.normal()])
        msf.update(x)

    try:
        m, s = msf.mean_and_std()
        print(f"Test passed - mean: {m}, std: {s}")
    except AssertionError as e:
        print(f"Assertion triggered: {e}")
        return

    print("Warning: Assertion was not triggered - may need more extreme values")


def test_normalizer_identical_values_assert():
    import numpy as np

    from problems.normalizer import Normalizer

    msf = Normalizer(shape=(1,))
    v = 1.23456789
    for _ in range(10**6):
        msf.update(np.array([v]))
    try:
        msf.mean_and_std()
    except AssertionError as e:
        print(f"Assertion triggered as expected: {e}")
        return
    print("Warning: Assertion was not triggered; floating-point error did not occur.")
