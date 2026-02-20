from common.telemetry import Telemetry


def test_telemetry_format():
    t = Telemetry()
    t.set_dt_fit(1.234)
    t.set_dt_select(5.6789)
    assert t.format() == "fit_dt=1.234 select_dt=5.679"


def test_telemetry_format_unset():
    t = Telemetry()
    assert t.format() == "fit_dt=N/A select_dt=N/A"


def test_telemetry_reset():
    t = Telemetry()
    t.set_dt_fit(1.0)
    t.set_dt_select(2.0)
    t.reset()
    assert t._dt_fit is None
    assert t._dt_select is None
