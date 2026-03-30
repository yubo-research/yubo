def test_morbo_enn_designer_builds():
    """Regression test: morbo-enn designer should build without crashing."""
    from optimizer.designer_registry import _build_turbo_enn, _SimpleContext

    ctx = _SimpleContext(
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
    _build_turbo_enn(ctx, "morbo-enn")
