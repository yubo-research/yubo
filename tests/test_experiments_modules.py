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


class TestFuncNames:
    def test_funcs_nd_is_list(self):
        from experiments.func_names import funcs_nd

        assert isinstance(funcs_nd, list)
        assert len(funcs_nd) > 0
        assert "ackley" in funcs_nd
        assert "sphere" in funcs_nd

    def test_funcs_1d_is_list(self):
        from experiments.func_names import funcs_1d

        assert isinstance(funcs_1d, list)
        assert len(funcs_1d) > 0

    def test_funcs_all_is_list(self):
        from experiments.func_names import funcs_all

        assert isinstance(funcs_all, list)
        assert len(funcs_all) > 0

    def test_func_brief_is_list(self):
        from experiments.func_names import func_brief

        assert isinstance(func_brief, list)
        assert len(func_brief) > 0

    def test_funcs_multimodal_contains_known_functions(self):
        from experiments.func_names import funcs_multimodal

        assert "ackley" in funcs_multimodal
        assert "griewank" in funcs_multimodal
        assert "rastrigin" in funcs_multimodal

    def test_funcs_bowl_contains_known_functions(self):
        from experiments.func_names import funcs_bowl

        assert "sphere" in funcs_bowl

    def test_funcs_valley_contains_known_functions(self):
        from experiments.func_names import funcs_valley

        assert "rosenbrock" in funcs_valley


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


class TestModalBatches:
    def test_job_key(self):
        from experiments.modal_batches import _job_key

        key = _job_key("test_batch", 42)
        assert key == "test_batch-42"

    @patch("experiments.modal_batches.prep_d_argss")
    @patch("experiments.modal_batches.mk_replicates")
    @patch("experiments.modal_batches.data_is_done")
    def test_gen_jobs(self, mock_data_is_done, mock_mk_replicates, mock_prep_d_argss):
        from experiments.modal_batches import _gen_jobs

        mock_data_is_done.return_value = False
        mock_run_config = MagicMock()
        mock_run_config.trace_fn = "/path/to/trace"
        mock_mk_replicates.return_value = [mock_run_config]
        mock_config = MagicMock()
        mock_prep_d_argss.return_value = [mock_config]

        jobs = list(_gen_jobs("test_batch"))

        assert len(jobs) == 1
        key, run_config = jobs[0]
        assert key == "test_batch-0"
        assert run_config == mock_run_config

    @patch("experiments.modal_batches.prep_d_argss")
    @patch("experiments.modal_batches.mk_replicates")
    @patch("experiments.modal_batches.data_is_done")
    def test_gen_jobs_skips_done(self, mock_data_is_done, mock_mk_replicates, mock_prep_d_argss):
        from experiments.modal_batches import _gen_jobs

        mock_data_is_done.return_value = True
        mock_run_config = MagicMock()
        mock_run_config.trace_fn = "/path/to/trace"
        mock_mk_replicates.return_value = [mock_run_config]
        mock_config = MagicMock()
        mock_prep_d_argss.return_value = [mock_config]

        jobs = list(_gen_jobs("test_batch"))

        assert len(jobs) == 0


class TestDistModal:
    def test_dist_modal_init(self):
        from experiments.dist_modal import DistModal

        dist = DistModal("test_app", "test_function", "/path/to/jobs")
        assert dist._app_name == "test_app"
        assert dist._function_name == "test_function"
        assert dist._job_fn == "/path/to/jobs"
