import time

import pytest
import torch
import torch.nn as nn

from common.collector import Collector
from uhd.uhd_collector import UHDCollector


def test_uhd_collector_basic():
    collector = Collector()
    uhd_collector = UHDCollector(
        name="tm_sphere", opt_name="turbo", collector=collector
    )
    assert uhd_collector.name == "tm_sphere"
    assert uhd_collector.opt_name == "turbo"


def test_uhd_collector_timing():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test_opt", collector=collector)

    uhd_collector.start_prop()
    time.sleep(0.01)
    dt_prop = uhd_collector.stop_prop()
    assert dt_prop >= 0.01

    uhd_collector.start_eval()
    time.sleep(0.005)
    dt_eval = uhd_collector.stop_eval()
    assert dt_eval >= 0.005


def test_uhd_collector_trace():
    collector = Collector()
    uhd_collector = UHDCollector(name="tm_mnist", opt_name="adamw", collector=collector)

    uhd_collector.start_prop()
    time.sleep(0.001)
    uhd_collector.stop_prop()

    uhd_collector.start_eval()
    time.sleep(0.0005)
    uhd_collector.stop_eval()

    uhd_collector.trace(y=0.1234)

    lines = list(collector)
    assert len(lines) == 1
    line = lines[0]
    assert "TRACE:" in line
    assert "name = tm_mnist" in line
    assert "opt_name = adamw" in line
    assert "i_iter = 0" in line
    assert "y = 0.1234" in line
    assert "y_best = 0.1234" in line
    assert "dt_prop" in line
    assert "dt_eval" in line


def test_uhd_collector_delegates_to_collector():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    uhd_collector("custom line 1")
    uhd_collector("custom line 2")

    lines = list(collector)
    assert len(lines) == 2
    assert lines[0] == "custom line 1"
    assert lines[1] == "custom line 2"


def test_uhd_collector_stop_without_start_raises():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    with pytest.raises(AssertionError, match="start_prop"):
        uhd_collector.stop_prop()

    with pytest.raises(AssertionError, match="start_eval"):
        uhd_collector.stop_eval()


def test_uhd_collector_default_collector():
    uhd_collector = UHDCollector(name="test", opt_name="test")
    assert isinstance(uhd_collector._collector, Collector)

    uhd_collector("test line")
    lines = list(uhd_collector._collector)
    assert len(lines) == 1
    assert lines[0] == "test line"


def test_uhd_collector_trace_formatting():
    collector = Collector()
    uhd_collector = UHDCollector(
        name="tm_ackley", opt_name="turbo", collector=collector
    )

    uhd_collector.start_prop()
    time.sleep(0.0015)
    uhd_collector.stop_prop()

    uhd_collector.start_eval()
    time.sleep(0.000023)
    uhd_collector.stop_eval()

    uhd_collector.trace(y=-1.2345)

    lines = list(collector)
    assert len(lines) == 1
    line = lines[0]
    assert "dt_prop" in line
    assert "dt_eval" in line
    assert "y = -1.2345" in line
    assert "y_best = -1.2345" in line


def test_uhd_collector_trace_without_stop_raises():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    with pytest.raises(AssertionError, match="stop_prop"):
        uhd_collector.trace(y=1.0)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()

    with pytest.raises(AssertionError, match="stop_eval"):
        uhd_collector.trace(y=1.0)


def test_uhd_collector_double_stop_raises():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()
    with pytest.raises(AssertionError, match="already called"):
        uhd_collector.stop_prop()

    uhd_collector.start_eval()
    uhd_collector.stop_eval()
    with pytest.raises(AssertionError, match="already called"):
        uhd_collector.stop_eval()


def test_uhd_collector_start_without_stop_prevents_new_start():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    uhd_collector.start_prop()
    with pytest.raises(AssertionError, match="already called"):
        uhd_collector.start_prop()
    uhd_collector.stop_prop()

    uhd_collector.start_eval()
    with pytest.raises(AssertionError, match="already called"):
        uhd_collector.start_eval()
    uhd_collector.stop_eval()


def test_uhd_collector_trace_resets_timings():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()
    uhd_collector.start_eval()
    uhd_collector.stop_eval()
    uhd_collector.trace(y=1.0)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()
    uhd_collector.start_eval()
    uhd_collector.stop_eval()
    uhd_collector.trace(y=2.0)

    lines = list(collector)
    assert len(lines) == 2
    assert "i_iter = 0" in lines[0]
    assert "i_iter = 1" in lines[1]
    assert "y_best = 1.0000" in lines[0]
    assert "y_best = 2.0000" in lines[1]


def test_uhd_collector_tracks_y_best():
    collector = Collector()
    uhd_collector = UHDCollector(name="test", opt_name="test", collector=collector)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()
    uhd_collector.start_eval()
    uhd_collector.stop_eval()
    uhd_collector.trace(y=1.0)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()
    uhd_collector.start_eval()
    uhd_collector.stop_eval()
    uhd_collector.trace(y=0.5)

    uhd_collector.start_prop()
    uhd_collector.stop_prop()
    uhd_collector.start_eval()
    uhd_collector.stop_eval()
    uhd_collector.trace(y=2.0)

    lines = list(collector)
    assert len(lines) == 3
    assert "i_iter = 0" in lines[0]
    assert "i_iter = 1" in lines[1]
    assert "i_iter = 2" in lines[2]
    assert "y_best = 1.0000" in lines[0]
    assert "y_best = 1.0000" in lines[1]
    assert "y_best = 2.0000" in lines[2]


def test_uhd_collector_params_logs_stats():
    collector = Collector()
    uhd_collector = UHDCollector(name="dummy", opt_name="opt", collector=collector)

    class Small(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.name = "small"
            self.weight = nn.Parameter(torch.tensor([[1.0, -1.0]], dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, self.weight.t())

    model = Small()
    uhd_collector.params(model)
    lines = list(collector)
    assert len(lines) == 1
    assert "PARAMS: target = small" in lines[0]
    assert "min = -1.000000" in lines[0]
    assert "max = 1.000000" in lines[0]
    assert "mean = 0.000000" in lines[0]
    assert "std = 1.000000" in lines[0]
