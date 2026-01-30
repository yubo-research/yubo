from unittest.mock import MagicMock, patch


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
        assert key == "test_batch-/path/to/trace"
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
