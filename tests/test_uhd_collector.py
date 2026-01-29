def test_uhd_collector_init():
    from uhd.uhd_collector import UHDCollector

    collector = UHDCollector(name="test", opt_name="test_opt")
    assert collector.name == "test"
    assert collector.opt_name == "test_opt"


def test_uhd_collector_update_best():
    from uhd.uhd_collector import UHDCollector

    collector = UHDCollector(name="test", opt_name="test_opt")
    collector.update_best(1.0)
    assert collector.best() == 1.0

    # Update with a better value
    collector.update_best(2.0)
    assert collector.best() == 2.0

    # Worse value should not update
    collector.update_best(1.5)
    assert collector.best() == 2.0


def test_uhd_collector_reset_eval_timing():
    from uhd.uhd_collector import UHDCollector

    collector = UHDCollector(name="test", opt_name="test_opt")
    collector.reset_eval_timing()
    # Should just reset without error


def test_uhd_collector_prop_timing():
    from uhd.uhd_collector import UHDCollector

    collector = UHDCollector(name="test", opt_name="test_opt")
    collector.start_prop()
    dt = collector.stop_prop()
    assert dt >= 0


def test_uhd_collector_eval_timing():
    from uhd.uhd_collector import UHDCollector

    collector = UHDCollector(name="test", opt_name="test_opt")
    collector.start_eval()
    dt = collector.stop_eval()
    assert dt >= 0
