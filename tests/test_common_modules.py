import numpy as np


class TestAllBounds:
    def test_constants(self):
        import common.all_bounds as all_bounds

        assert all_bounds.x_low == -1.0
        assert all_bounds.x_high == 1.0
        assert all_bounds.x_width == 2.0

        assert all_bounds.p_low == -1.0
        assert all_bounds.p_high == 1.0
        assert all_bounds.p_width == 2.0

        assert all_bounds.bt_low == 0.0
        assert all_bounds.bt_high == 1.0
        assert all_bounds.bt_width == 1.0

    def test_get_box_bounds_x(self):
        import common.all_bounds as all_bounds

        box = all_bounds.get_box_bounds_x(3)
        assert box.shape == (3,)
        np.testing.assert_array_equal(box.low, [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(box.high, [1.0, 1.0, 1.0])

    def test_get_box_bounds_x_1d(self):
        import common.all_bounds as all_bounds

        box = all_bounds.get_box_bounds_x(1)
        assert box.shape == (1,)
        np.testing.assert_array_equal(box.low, [-1.0])
        np.testing.assert_array_equal(box.high, [1.0])

    def test_get_box_1d01(self):
        import common.all_bounds as all_bounds

        box = all_bounds.get_box_1d01()
        assert box.low == 0.0
        assert box.high == 1.0


class TestSeedAll:
    def test_seed_all_deterministic(self):
        from common.seed_all import seed_all

        seed_all(42)
        vals1 = [np.random.rand() for _ in range(5)]

        seed_all(42)
        vals2 = [np.random.rand() for _ in range(5)]

        np.testing.assert_array_almost_equal(vals1, vals2)

    def test_seed_all_different_seeds(self):
        from common.seed_all import seed_all

        seed_all(42)
        vals1 = np.random.rand()

        seed_all(123)
        vals2 = np.random.rand()

        assert vals1 != vals2

    def test_seed_all_torch_deterministic(self):
        import torch

        from common.seed_all import seed_all

        seed_all(42)
        t1 = torch.rand(3)

        seed_all(42)
        t2 = torch.rand(3)

        torch.testing.assert_close(t1, t2)


class TestParseKv:
    def test_parse_kv_simple(self):
        from common.util import parse_kv

        result = parse_kv(["a=1", "b=2"])
        assert result == {"a": "1", "b": "2"}

    def test_parse_kv_empty(self):
        from common.util import parse_kv

        result = parse_kv([])
        assert result == {}

    def test_parse_kv_with_dashes(self):
        from common.util import parse_kv

        result = parse_kv(["--opt-name=random", "--num-arms=5"])
        assert result == {"--opt-name": "random", "--num-arms": "5"}


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
        from common.telemetry import Telemetry

        t = Telemetry()
        t.set_dt_fit(1.0)
        t.set_dt_select(2.0)
        t.reset()
        assert t._dt_fit is None
        assert t._dt_select is None

    def test_format_with_values(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        t.set_dt_fit(1.234)
        t.set_dt_select(5.6789)
        assert t.format() == "fit_dt=1.234 select_dt=5.679"

    def test_format_with_none(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        assert t.format() == "fit_dt=N/A select_dt=N/A"

    def test_format_partial(self):
        from common.telemetry import Telemetry

        t = Telemetry()
        t.set_dt_fit(1.0)
        assert t.format() == "fit_dt=1.000 select_dt=N/A"

        t.reset()
        t.set_dt_select(2.0)
        assert t.format() == "fit_dt=N/A select_dt=2.000"
