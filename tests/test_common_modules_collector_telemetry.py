class TestCollector:
    def test_init(self):
        from collections import deque

        from common.collector import Collector

        c = Collector()
        assert c._lines == deque()

    def test_call_adds_line(self, capsys):
        from collections import deque

        from common.collector import Collector

        c = Collector()
        c("hello")
        assert c._lines == deque(["hello"])

        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_iteration(self):
        from common.collector import Collector

        c = Collector()
        c("a")
        c("b")
        c("c")

        result = list(c)
        assert result == ["a", "b", "c"]

    def test_multiple_iterations(self):
        from common.collector import Collector

        c = Collector()
        c("a")
        c("b")

        result1 = list(c)
        result2 = list(c)
        assert result1 == result2 == ["a", "b"]


class TestTelemetry:
    def test_init(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        assert t._dt_fit is None
        assert t._dt_select is None

    def test_set_dt_fit(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        t.set_dt_fit(1.5)
        assert t._dt_fit == 1.5

    def test_set_dt_select(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        t.set_dt_select(2.5)
        assert t._dt_select == 2.5

    def test_reset(self):
        from tests.test_util import assert_telemetry_reset_clears_dt_fields

        assert_telemetry_reset_clears_dt_fields()

    def test_format_with_values(self):
        from tests.test_util import assert_telemetry_format_fit_select_values

        assert_telemetry_format_fit_select_values()

    def test_format_with_none(self):
        from tests.test_util import assert_telemetry_format_all_na

        assert_telemetry_format_all_na()

    def test_format_partial(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        t.set_dt_fit(1.0)
        assert t.format() == "fit_dt = 1.000 select_dt = N/A"

        t.reset()
        t.set_dt_select(2.0)
        assert t.format() == "fit_dt = N/A select_dt = 2.000"
