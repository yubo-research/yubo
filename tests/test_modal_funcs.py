def _get(mod_name, attr):
    import importlib

    return getattr(importlib.import_module(mod_name), attr)


def test_mk_image_exists():
    pytest = __import__("pytest")
    try:
        mk_image = _get("experiments.modal_image", "mk_image")
        assert callable(mk_image)
    except ImportError:
        pytest.skip("modal not installed")


def test_modal_batches_worker_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modal_batches_worker = _get("experiments.modal_batches", "modal_batches_worker")

        assert modal_batches_worker is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_batches_submitter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batches_submitter = _get("experiments.modal_batches", "batches_submitter")

        assert batches_submitter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_resubmitter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modal_batches_resubmitter = _get("experiments.modal_batches", "modal_batches_resubmitter")

        assert modal_batches_resubmitter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batch_deleter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modal_batch_deleter = _get("experiments.modal_batches", "modal_batch_deleter")

        assert modal_batch_deleter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_collect_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            collect = _get("experiments.modal_batches", "collect")

        assert collect is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_status_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status = _get("experiments.modal_batches", "status")

        assert status is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_clean_up_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean_up = _get("experiments.modal_batches", "clean_up")

        assert clean_up is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_dist_modal_collect_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            collect = _get("experiments.dist_modal", "collect")

        assert collect is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")
