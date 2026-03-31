import pytest

from optimizer.designer_errors import NoSuchDesignerError


def _make_ctx():
    from optimizer.designer_registry import _SimpleContext

    return _SimpleContext(
        policy=None,
        num_arms=1,
        bt=None,
        num_keep=1,
        keep_style="best",
        num_keep_val=1,
        init_yubo_default=10,
        init_ax_default=10,
        default_num_X_samples=100,
    )


def test_morbo_enn_designer_builds():
    """Regression test: morbo-enn designer should build without crashing."""
    from optimizer.designer_registry import _build_turbo_enn

    ctx = _make_ctx()
    _build_turbo_enn(ctx, "morbo-enn")


@pytest.mark.parametrize("kind", ["turbo-one-nds", "turbo-one-ucb"])
def test_turbo_one_acq_variants_build(kind):
    """Test that turbo-one-nds and turbo-one-ucb designers build correctly."""
    from optimizer.designer_registry import _build_turbo_enn

    ctx = _make_ctx()
    designer = _build_turbo_enn(ctx, kind)
    assert designer is not None


def test_d_turbo_enn_fit_ucb_defaults():
    from optimizer.designer_registry import _d_turbo_enn_fit_ucb

    ctx = _make_ctx()
    d = _d_turbo_enn_fit_ucb(ctx, {})
    assert d._k == 10
    assert d._num_fit_samples == 100
    assert d._num_fit_candidates == 100


def test_d_turbo_enn_fit_ucb_nfs_and_k():
    from optimizer.designer_registry import _d_turbo_enn_fit_ucb

    ctx = _make_ctx()
    d = _d_turbo_enn_fit_ucb(ctx, {"nfs": 50, "k": 7})
    assert d._k == 7
    assert d._num_fit_samples == 50


def test_d_turbo_enn_fit_ucb_rejects_unknown_option():
    from optimizer.designer_registry import _d_turbo_enn_fit_ucb

    ctx = _make_ctx()
    with pytest.raises(NoSuchDesignerError, match="does not support"):
        _d_turbo_enn_fit_ucb(ctx, {"bogus": 1})
