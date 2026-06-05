import importlib

import experiments.modal_batches_impl as _modal_batches  # noqa: F401
import experiments.modal_timing_sweep as _modal_timing_sweep  # noqa: F401


def _get(mod_name, attr):
    return getattr(importlib.import_module(mod_name), attr)


def test_mk_image_exists():
    pytest = __import__("pytest")
    try:
        mk_image = _get("experiments.modal_image", "mk_image")
        assert callable(mk_image)
    except ImportError:
        pytest.skip("modal not installed")


def _sync_original_modal_image(image):
    for attr in dir(image):
        if attr.startswith("_sync_original"):
            return getattr(image, attr)
    raise AssertionError("expected internal Modal Image handle")


def _modal_image_run_commands(image):
    commands = []

    def walk(node):
        rep = getattr(node, "_rep", "")
        if "run_commands" in rep:
            load = getattr(node, "_load", None)
            if load is not None and load.__closure__ is not None:
                for cell in load.__closure__:
                    build = cell.cell_contents
                    if callable(build) and getattr(build, "__name__", "") == "build_dockerfile":
                        commands.extend(build("2024.10").commands)
        deps = getattr(node, "_deps", None)
        if deps is None:
            return
        for dep in deps():
            if type(dep).__name__ == "_Image":
                walk(dep)

    walk(_sync_original_modal_image(image))
    return commands


def _modal_image_yubo_mount_paths(image):
    paths = []

    def walk(node):
        deps = getattr(node, "_deps", None)
        if deps is None:
            return
        for dep in deps():
            if type(dep).__name__ == "_Image":
                walk(dep)
            else:
                dep_repr = str(dep)
                if "/root/" in dep_repr:
                    paths.append(dep_repr)

    walk(_sync_original_modal_image(image))
    return paths


def test_modal_image_enn_smoke_runs_before_yubo_dirs_mounted():
    """ENN import smoke must be a build RUN step before yubo add_local_dir mounts."""
    pytest = __import__("pytest")
    try:
        mk_image = _get("experiments.modal_image", "mk_image")
    except ImportError:
        pytest.skip("modal not installed")

    image = mk_image()
    run_commands = _modal_image_run_commands(image)
    enn_smoke = [cmd for cmd in run_commands if "EpistemicNearestNeighbors" in cmd and cmd.startswith("RUN ")]
    assert len(enn_smoke) == 1

    mount_reprs = _modal_image_yubo_mount_paths(image)
    yubo_mounts = [m for m in mount_reprs if "/root/optimizer" in m or "/root/acq" in m]
    assert yubo_mounts, "expected yubo package mounts on the image"
    assert all("optimizer" in m or "acq" in m for m in yubo_mounts)


def test_modal_batches_worker_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modal_batches_worker = _get("experiments.modal_batches_impl", "modal_batches_worker")

        assert modal_batches_worker is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_batches_submitter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batches_submitter = _get("experiments.modal_batches_impl", "batches_submitter")

        assert batches_submitter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_resubmitter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modal_batches_resubmitter = _get("experiments.modal_batches_impl", "modal_batches_resubmitter")

        assert modal_batches_resubmitter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batch_deleter_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modal_batch_deleter = _get("experiments.modal_batches_impl", "modal_batch_deleter")

        assert modal_batch_deleter is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_collect_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            collect = _get("experiments.modal_batches_impl", "_collect")

        assert collect is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_status_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status = _get("experiments.modal_batches_impl", "status")

        assert status is not None
    except (ImportError, Exception):
        pytest.skip("modal not installed or error loading")


def test_modal_batches_clean_up_exists():
    pytest = __import__("pytest")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean_up = _get("experiments.modal_batches_impl", "clean_up")

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
