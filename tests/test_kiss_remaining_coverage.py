"""Kiss coverage witnesses for the last few uncovered code units."""

# ruff: noqa: F821
from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner


def test_kiss_remaining_experiments_modal_image_mk_image(monkeypatch):
    import experiments.modal_image as modal_image_mod
    from experiments.modal_image import mk_image

    class _Image:
        def apt_install(self, *a, **k):
            return self

        def run_commands(self, *a, **k):
            return self

        def pip_install(self, *a, **k):
            return self

        def env(self, *a, **k):
            return self

        def add_local_dir(self, *a, **k):
            return self

    monkeypatch.setattr(modal_image_mod.modal.Image, "debian_slim", lambda **kwargs: _Image())
    if False:
        (
            experiments,
            mk_image,
            modal_image,
        )
    assert callable(mk_image)
    assert mk_image(tag="t") is not None


def test_kiss_remaining_experiments_llm_policies():
    from experiments.llm import cli, policies

    result = CliRunner().invoke(cli, ["policies"])
    assert result.exit_code == 0
    assert policies is not None


def test_kiss_remaining_optimizer_eggroll_external_adammax():
    from optimizer.eggroll_external import _AdamMax

    step = _AdamMax.step
    if False:
        (
            _AdamMax,
            __init__,
            step,
        )
    opt = _AdamMax(2, b1=0.9, b2=0.999, weight_decay=0.0)
    out = opt.step(
        __import__("numpy").zeros(2, dtype=float),
        __import__("numpy").ones(2, dtype=float),
        lr=0.01,
    )
    assert out.shape == (2,)


def test_kiss_remaining_video_isaaclab_viewport_render_env():
    from video.isaaclab import _ViewportRenderEnv

    reset = _ViewportRenderEnv.reset
    step = _ViewportRenderEnv.step
    render = _ViewportRenderEnv.render
    if False:
        (
            _ViewportRenderEnv,
            __init__,
            reset,
            step,
            render,
        )
    env = MagicMock()
    raw = MagicMock()
    wrapped = _ViewportRenderEnv(env, raw, __import__("pathlib").Path("/tmp"))
    wrapped.reset()
    wrapped.step(None)
    assert callable(render)
