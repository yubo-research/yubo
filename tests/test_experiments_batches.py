import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestExperimentUtil:
    def test_ensure_parent_creates_directory(self):
        from experiments.experiment_util import ensure_parent

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = f"{tmpdir}/subdir1/subdir2/file.txt"
            ensure_parent(nested_path)
            assert os.path.isdir(f"{tmpdir}/subdir1/subdir2")

    def test_ensure_parent_existing_directory(self):
        from experiments.experiment_util import ensure_parent

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = f"{tmpdir}/file.txt"
            ensure_parent(nested_path)
            assert os.path.isdir(tmpdir)


class TestBatches:
    def test_prep_d_argss_valid_batch(self):
        from experiments.batches import prep_d_argss

        result = prep_d_argss("prep_cum_time_dim")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_prep_d_argss_invalid_batch(self):
        from experiments.batches import prep_d_argss

        with pytest.raises(AssertionError):
            prep_d_argss("nonexistent_batch_tag")

    def test_worker_returns_exit_code(self):
        from experiments.batches import worker

        result = worker("true")
        assert result == 0

        result = worker("false")
        assert result != 0

    @patch("experiments.batches.multiprocessing")
    @patch("experiments.batches.os")
    def test_run_batch_dry_run(self, mock_os, mock_mp):
        from experiments.batches import run_batch
        from experiments.experiment_sampler import ExperimentConfig

        mock_os.makedirs = MagicMock()

        config = ExperimentConfig(
            exp_dir="/tmp/test",
            env_tag="f:sphere-2d",
            opt_name="random",
            num_arms=1,
            num_rounds=1,
            num_reps=1,
        )
        d_argss = [config.to_dict()]

        run_batch(d_argss, b_dry_run=True)

        mock_mp.Process.assert_not_called()


class TestBatchPreps:
    def test_prep_cum_time_dim(self):
        from experiments.batch_preps import prep_cum_time_dim
        from experiments.experiment_sampler import ExperimentConfig

        result = prep_cum_time_dim("results")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(r, ExperimentConfig) for r in result)

    def test_prep_ts_sweep(self):
        from experiments.batch_preps import prep_ts_sweep
        from experiments.experiment_sampler import ExperimentConfig

        result = prep_ts_sweep("results")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(r, ExperimentConfig) for r in result)

    def test_prep_turbo_ackley_repro_returns_none(self):
        from experiments.batch_preps import prep_turbo_ackley_repro

        result = prep_turbo_ackley_repro("results")
        assert result is None
