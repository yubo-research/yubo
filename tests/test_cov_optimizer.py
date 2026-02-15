from optimizer.designer_registry import _SimpleContext


def test_simple_context_init():
    ctx = _SimpleContext(
        policy=None,
        num_arms=5,
        bt=None,
        num_keep=3,
        keep_style="best",
        num_keep_val=2,
        init_yubo_default=True,
        init_ax_default=False,
        default_num_X_samples=100,
    )
    assert ctx.num_arms == 5
    assert ctx.policy is None
    assert ctx.bt is None
    assert ctx.num_keep == 3
    assert ctx.keep_style == "best"
    assert ctx.num_keep_val == 2
    assert ctx.init_yubo_default is True
    assert ctx.init_ax_default is False
    assert ctx.default_num_X_samples == 100
