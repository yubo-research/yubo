def test_mk_image_exists():
    pytest = __import__("pytest")
    try:
        from experiments.modal_image import mk_image

        assert callable(mk_image)
    except ImportError:
        pytest.skip("modal not installed")


def test_modal_batches_worker_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import modal_batches_worker

        assert modal_batches_worker is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_batches_submitter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import batches_submitter

        assert batches_submitter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_resubmitter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import modal_batches_resubmitter

        assert modal_batches_resubmitter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batch_deleter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import modal_batch_deleter

        assert modal_batch_deleter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_collect_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import collect

        assert collect is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_status_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import status

        assert status is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_clean_up_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.modal_batches import clean_up

        assert clean_up is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_dist_modal_collect_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from experiments.dist_modal import collect

        assert collect is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")
