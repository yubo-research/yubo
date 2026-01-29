import pytest


def test_turbo_restart_error():
    from acq.turbo_yubo.ty_exceptions import TuRBORestartError

    with pytest.raises(TuRBORestartError):
        raise TuRBORestartError("test")


def test_turbo_restart_error_is_exception():
    from acq.turbo_yubo.ty_exceptions import TuRBORestartError

    assert issubclass(TuRBORestartError, Exception)
