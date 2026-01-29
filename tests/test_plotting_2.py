import json
import os
import tempfile


def test_infer_experiment_from_configs():
    from analysis.plotting_2 import infer_experiment_from_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create exp_dir structure
        exp_dir = "test_exp"
        os.makedirs(os.path.join(tmpdir, exp_dir, "run1"))

        # Create a config.json file
        config = {
            "env_tag": "ackley",
            "opt_name": "random",
            "num_arms": 5,
            "num_rounds": 10,
            "num_reps": 3,
        }
        with open(os.path.join(tmpdir, exp_dir, "run1", "config.json"), "w") as f:
            json.dump(config, f)

        result = infer_experiment_from_configs(tmpdir, exp_dir)
        assert "env_tags" in result
        assert "opt_names" in result
        assert "ackley" in result["env_tags"]
        assert "random" in result["opt_names"]


def test_infer_experiment_from_configs_no_configs():
    import pytest

    from analysis.plotting_2 import infer_experiment_from_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = "empty_exp"
        os.makedirs(os.path.join(tmpdir, exp_dir))

        with pytest.raises(ValueError):
            infer_experiment_from_configs(tmpdir, exp_dir)


def test_infer_experiment_from_configs_not_found():
    import pytest

    from analysis.plotting_2 import infer_experiment_from_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            infer_experiment_from_configs(tmpdir, "nonexistent_exp")
