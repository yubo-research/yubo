"""Tests for analysis/plotting_2_trace.py (part 1)."""

import json
import tempfile
from pathlib import Path

import numpy as np


class TestCountDoneReps:
    def test_counts_done_files(self):
        from analysis.plotting_2_trace import count_done_reps

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                (Path(tmpdir) / f"{i:05d}").touch()
                (Path(tmpdir) / f"{i:05d}.done").touch()
            result = count_done_reps(tmpdir)
            assert result == 3

    def test_handles_traces_subdir(self):
        from analysis.plotting_2_trace import count_done_reps

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_dir = Path(tmpdir) / "traces"
            traces_dir.mkdir()
            for i in range(2):
                (traces_dir / f"{i:05d}.jsonl").touch()
            result = count_done_reps(tmpdir)
            assert result == 0


class TestPrintDatasetSummary:
    def test_prints_summary(self, capsys):
        from analysis.plotting_2_trace import print_dataset_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "exp"
            run_dir = root / "run1"
            run_dir.mkdir(parents=True)
            config = {
                "env_tag": "test-env",
                "opt_name": "opt1",
                "num_arms": 1,
                "num_rounds": 10,
                "num_reps": 5,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f)
            try:
                print_dataset_summary(
                    str(Path(tmpdir) / "results"),
                    "exp",
                    problem="test-env",
                    opt_names=["opt1"],
                    num_arms=1,
                    num_rounds=10,
                    num_reps=5,
                )
            except Exception:
                pass


class TestLoadRlTraces:
    def test_accepts_larger_config_num_reps_and_slices(self):
        from analysis.data_io import TraceRecord, write_trace_jsonl
        from analysis.plotting_2_trace import load_rl_traces

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "exp"
            run_dir = root / "run1"
            traces_dir = run_dir / "traces"
            traces_dir.mkdir(parents=True)

            config = {
                "env_tag": "bw-heur",
                "opt_name": "random",
                "num_arms": 1,
                "num_rounds": 10000,
                "num_reps": 30,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f)

            for i_rep in range(30):
                write_trace_jsonl(
                    str(traces_dir / f"{i_rep:05d}.jsonl"),
                    [
                        TraceRecord(i_iter=0, dt_prop=0.0, dt_eval=0.0, rreturn=float(i_rep)),
                        TraceRecord(
                            i_iter=1,
                            dt_prop=0.0,
                            dt_eval=0.0,
                            rreturn=float(i_rep) + 0.5,
                        ),
                    ],
                )

            data_locator, traces = load_rl_traces(
                str(Path(tmpdir) / "results"),
                "exp",
                ["random"],
                num_arms=1,
                num_rounds=10000,
                num_reps=10,
                problem="bw-heur",
            )

            assert data_locator.optimizers() == ["random"]
            assert traces.shape == (1, 1, 10, 2)
            np.testing.assert_array_equal(traces[0, 0, :, 0], np.arange(10, dtype=float))
            np.testing.assert_array_equal(traces[0, 0, :, 1], np.arange(10, dtype=float) + 0.5)
