from pathlib import Path


def test_fan_in_out_check_passes_on_repo():
    from admin.fan_in_out import check_fan_limits

    repo_root = Path(__file__).resolve().parents[1]
    assert check_fan_limits(repo_root) == []
