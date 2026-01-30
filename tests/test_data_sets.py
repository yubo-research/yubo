import os
import tempfile

import numpy as np


def test_extract_kv():
    from analysis.data_sets import extract_kv

    x = ["a", "=", "1", "b", "=", "2"]
    result = extract_kv(x)
    assert result == {"a": "1", "b": "2"}


def test_load_basic():
    from analysis.data_sets import load

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("x = 1.0 y = 2.0\n")
        f.write("x = 3.0 y = 4.0\n")
        temp_path = f.name

    try:
        data = load(temp_path, ["x", "y"])
        assert data.shape == (2, 2)
        assert np.isclose(data[0, 0], 1.0)
        assert np.isclose(data[1, 1], 4.0)
    finally:
        os.unlink(temp_path)


def test_load_kv():
    from analysis.data_sets import load_kv

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("x = 1.0 y = 2.0\n")
        f.write("x = 3.0 y = 4.0\n")
        temp_path = f.name

    try:
        data = load_kv(temp_path, ["x", "y"])
        assert "x" in data
        assert "y" in data
        assert len(data["x"]) == 2
    finally:
        os.unlink(temp_path)


def test_load_traces_jsonl():
    import json
    from pathlib import Path

    from analysis.data_sets import load_traces_jsonl

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a trace file in the directory
        trace_file = Path(tmpdir) / "00000.jsonl"
        with open(trace_file, "w") as f:
            f.write(json.dumps({"i_iter": 0, "dt_prop": 0.1, "dt_eval": 0.2, "rreturn": 1.0}) + "\n")
            f.write(json.dumps({"i_iter": 1, "dt_prop": 0.2, "dt_eval": 0.3, "rreturn": 2.0}) + "\n")

        # Create a .done marker file
        done_file = Path(tmpdir) / "00000.done"
        done_file.touch()

        traces = load_traces_jsonl(tmpdir)
        assert traces.shape[0] == 1  # one replication
        assert traces.shape[1] == 2  # two iterations


def test_cache_stats():
    from analysis.data_sets import cache_stats

    stats = cache_stats()
    assert "load_kv" in stats
    assert "load_traces_jsonl" in stats
    assert "load_traces" in stats


def test_clear_cache():
    from analysis.data_sets import clear_cache

    # Just verify it runs without error
    clear_cache()


def test_range_summarize():
    from analysis.data_sets import range_summarize

    # traces[i_problem, i_opt, i_replication, i_round]
    traces = np.random.rand(2, 3, 2, 4)
    mu, se = range_summarize(traces)
    assert mu.shape == (3,)
    assert se.shape == (3,)


def test_rank_summarize():
    from analysis.data_sets import rank_summarize

    # Create test data: 2 problems, 3 optimizers, 2 replications, 4 rounds
    traces = np.random.rand(2, 3, 2, 4)
    mu, se = rank_summarize(traces)
    assert mu.shape == (3,)
    assert se.shape == (3,)


def test_problems_in():
    from analysis.data_sets import problems_in

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some problem directories
        os.makedirs(os.path.join(tmpdir, "exp1", "problem1"))
        os.makedirs(os.path.join(tmpdir, "exp1", "problem2"))

        problems = problems_in(tmpdir, "exp1")
        assert "problem1" in problems
        assert "problem2" in problems


def test_optimizers_in():
    from analysis.data_sets import optimizers_in

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create problem/optimizer directories
        os.makedirs(os.path.join(tmpdir, "exp1", "problem1", "opt1"))
        os.makedirs(os.path.join(tmpdir, "exp1", "problem1", "opt2"))

        optimizers = optimizers_in(tmpdir, "exp1", "problem1")
        assert "opt1" in optimizers
        assert "opt2" in optimizers


def test_all_in():
    from analysis.data_sets import all_in

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create problem/optimizer directories
        os.makedirs(os.path.join(tmpdir, "exp1", "problem1", "opt1"))
        os.makedirs(os.path.join(tmpdir, "exp1", "problem1", "opt2"))

        problems, optimizers = all_in(tmpdir, "exp1")
        assert "problem1" in problems
        assert "opt1" in optimizers
        assert "opt2" in optimizers
