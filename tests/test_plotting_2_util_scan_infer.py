"""Tests for analysis/plotting_2_util (scan_experiment_configs, infer_params_from_configs)."""

import json
import tempfile
from pathlib import Path


class TestScanExperimentConfigs:
    def test_returns_empty_for_nonexistent_dir(self):
        from analysis.plotting_2_util import scan_experiment_configs

        env_tags, opt_names = scan_experiment_configs(Path("/nonexistent/path"))
        assert env_tags == set()
        assert opt_names == set()

    def test_scans_configs(self):
        from analysis.plotting_2_util import scan_experiment_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subdir = root / "exp1"
            subdir.mkdir()
            config = {"env_tag": "test-env", "opt_name": "test-opt"}
            with open(subdir / "config.json", "w") as f:
                json.dump(config, f)
            env_tags, opt_names = scan_experiment_configs(root)
            assert "test-env" in env_tags
            assert "test-opt" in opt_names


class TestInferParamsFromConfigs:
    def test_infers_params(self):
        from analysis.plotting_2_util import infer_params_from_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "exp"
            root.mkdir(parents=True)
            for i, (env, arms, rounds, reps) in enumerate(
                [
                    ("prob_seq", 1, 100, 10),
                    ("prob_batch", 50, 30, 10),
                ]
            ):
                subdir = root / f"run{i}"
                subdir.mkdir()
                config = {
                    "env_tag": env,
                    "opt_name": "opt1",
                    "num_arms": arms,
                    "num_rounds": rounds,
                    "num_reps": reps,
                }
                with open(subdir / "config.json", "w") as f:
                    json.dump(config, f)
            result = infer_params_from_configs(
                str(Path(tmpdir) / "results"),
                "exp",
                problem_seq="prob_seq",
                problem_batch="prob_batch",
                opt_names=["opt1"],
            )
            assert result.get("num_arms_seq") == 1
            assert result.get("num_arms_batch") == 50
