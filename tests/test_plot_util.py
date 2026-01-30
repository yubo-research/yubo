import tempfile
from pathlib import Path

from analysis.plot_util import (
    collect_config_rows,
    normalize_results_and_exp_dir,
    uniq_int,
)


def test_uniq_int_single():
    assert uniq_int([1, 1, 1]) == 1


def test_uniq_int_none():
    assert uniq_int([1, 2, 3]) is None
    assert uniq_int([]) is None
    assert uniq_int(["a", "b"]) is None


def test_normalize_results_and_exp_dir_basic():
    rp, ed = normalize_results_and_exp_dir("/tmp/results", "exp1")
    assert ed == "exp1"


def test_collect_config_rows_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rows = collect_config_rows(root, ["opt1"], include_opt_name=True)
        assert rows == []
