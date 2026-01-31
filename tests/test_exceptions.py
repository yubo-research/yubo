import pytest


def test_wrong_dimensions_exception():
    from problems.exceptions import WrongDimensions

    with pytest.raises(WrongDimensions):
        raise WrongDimensions("test message")


def test_wrong_dimensions_is_exception():
    from problems.exceptions import WrongDimensions

    assert issubclass(WrongDimensions, Exception)
