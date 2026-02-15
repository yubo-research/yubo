from pathlib import Path

import modal

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_DIRS = ("ops", "optimizer", "problems", "common", "sampling")

_image = (
    modal.Image.debian_slim(python_version="3.11.9")
    .apt_install("swig")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.4",
        "gymnasium==1.2.0",
        "gymnasium[mujoco]",
        "gymnasium[box2d]",
        "mujoco==3.3.3",
        "scipy==1.15.3",
        "click==8.3.1",
    )
    .env({"PYTHONPATH": "/root"})
)
for _d in _PROJECT_DIRS:
    _image = _image.add_local_dir(str(_PROJECT_ROOT / _d), remote_path=f"/root/{_d}")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def run(
    env_tag,
    num_rounds,
    lr,
    ndt,
    nmt,
    *,
    gpu="A100",
    log_interval: int = 10,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
):
    app = modal.App(name="yubo-uhd")

    @app.function(image=_image, timeout=2 * 60 * 60, gpu=gpu, serialized=True)
    def run_uhd(
        env_tag,
        num_rounds,
        lr,
        sigma,
        ndt,
        nmt,
        log_interval,
        accuracy_interval,
        target_accuracy,
    ):
        import io
        import sys

        from ops.uhd_setup import make_loop

        loop = make_loop(
            env_tag,
            num_rounds,
            lr=lr,
            sigma=sigma,
            num_dim_target=ndt,
            num_module_target=nmt,
            log_interval=log_interval,
            accuracy_interval=accuracy_interval,
            target_accuracy=target_accuracy,
        )
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, buf)
        try:
            loop.run()
        finally:
            sys.stdout = real_stdout
        return buf.getvalue()

    with modal.enable_output():
        with app.run():
            return run_uhd.remote(
                env_tag,
                num_rounds,
                lr,
                0.001,
                ndt,
                nmt,
                int(log_interval),
                int(accuracy_interval),
                target_accuracy,
            )
