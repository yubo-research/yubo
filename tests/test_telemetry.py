from tests.test_util import (
    assert_telemetry_format_all_na,
    assert_telemetry_format_fit_select_values,
    assert_telemetry_reset_clears_dt_fields,
)


def test_telemetry_format():
    assert_telemetry_format_fit_select_values()


def test_telemetry_format_unset():
    assert_telemetry_format_all_na()


def test_telemetry_reset():
    assert_telemetry_reset_clears_dt_fields()
