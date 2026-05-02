"""Tests for analysis/plotting_2_util (noise_label, speedup_x_label)."""


class TestNoiseLabel:
    def test_frozen_noise(self):
        from analysis.plotting_2_util import noise_label

        assert noise_label("problem:fn") == "Frozen noise"

    def test_natural_noise(self):
        from analysis.plotting_2_util import noise_label

        assert noise_label("problem") == "Natural noise"


class TestSpeedupXLabel:
    def test_returns_none_when_no_data(self):
        from analysis.plotting_2_util import speedup_x_label

        assert speedup_x_label(None, "problem") is None
        assert speedup_x_label({}, "problem") is None

    def test_returns_none_when_no_baseline(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"other-opt": 10.0}, "problem")
        assert result is None

    def test_returns_none_when_no_compare_opt(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"turbo-one": 10.0}, "problem")
        assert result is None

    def test_computes_speedup_for_frozen_noise(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"turbo-one": 100.0, "turbo-enn-p": 10.0}, "problem:fn")
        assert result == "10x speedup"

    def test_computes_speedup_for_natural_noise(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"turbo-one": 50.0, "turbo-enn-fit/acq_type=ucb": 10.0}, "problem")
        assert result == "5x speedup"
