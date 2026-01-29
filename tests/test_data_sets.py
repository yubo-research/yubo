import os
import tempfile


def test_clear_cache():
    from analysis.data_sets import clear_cache

    clear_cache()


def test_cache_stats():
    from analysis.data_sets import cache_stats

    stats = cache_stats()
    assert "load_kv" in stats
    assert "load_traces_jsonl" in stats
    assert "load_traces" in stats


def test_extract_kv():
    from analysis.data_sets import extract_kv

    x = ["key", "=", "value", "num", "=", "42"]
    result = extract_kv(x)
    assert result == {"key": "value", "num": "42"}


def test_extract_kv_empty():
    from analysis.data_sets import extract_kv

    x = []
    result = extract_kv(x)
    assert result == {}


def test_problems_in():
    from analysis.data_sets import problems_in

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "exp_tag", "problem1"))
        os.makedirs(os.path.join(tmpdir, "exp_tag", "problem2"))
        problems = problems_in(tmpdir, "exp_tag")
        assert "problem1" in problems
        assert "problem2" in problems


def test_optimizers_in():
    from analysis.data_sets import optimizers_in

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "exp_tag", "problem1", "opt1"))
        os.makedirs(os.path.join(tmpdir, "exp_tag", "problem1", "opt2"))
        opts = optimizers_in(tmpdir, "exp_tag", "problem1")
        assert "opt1" in opts
        assert "opt2" in opts


def test_all_in():
    from analysis.data_sets import all_in

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "exp_tag", "problem1", "opt1"))
        os.makedirs(os.path.join(tmpdir, "exp_tag", "problem2", "opt2"))
        problems, optimizers = all_in(tmpdir, "exp_tag")
        assert "problem1" in problems
        assert "problem2" in problems
