from pathlib import Path

from admin.fan_in_out import FanStats, compute_fan_stats, main


def test_fan_stats():
    fs = FanStats(fan_in=3, fan_out=5)
    assert fs.fan_in == 3
    assert fs.fan_out == 5


def test_compute_fan_stats():
    repo_root = Path(__file__).resolve().parents[1]
    stats = compute_fan_stats(repo_root)
    assert isinstance(stats, dict)
    assert len(stats) > 0
    for mod, stat in stats.items():
        assert isinstance(mod, str)
        assert isinstance(stat, FanStats)
        assert isinstance(stat.fan_in, int)
        assert isinstance(stat.fan_out, int)


def test_main():
    result = main()
    assert isinstance(result, int)
    assert result in (0, 1)
