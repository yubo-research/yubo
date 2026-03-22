from optimizer.designer_registry import (
    _DESIGNER_DISPATCH,
    _DESIGNER_OPTION_SPECS,
    _SIMPLE_BUILDERS,
    _SimpleContext,
)


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


def test_optioned_designer_names_have_dispatch_entries():
    # Prevent drift between option metadata and actual dispatch table.
    optioned_names = set(_DESIGNER_OPTION_SPECS)
    dispatch_names = set(_DESIGNER_DISPATCH)
    assert optioned_names <= dispatch_names


def test_optioned_designers_are_not_declared_as_simple_builders():
    optioned_names = set(_DESIGNER_OPTION_SPECS)
    simple_names = set(_SIMPLE_BUILDERS)
    assert optioned_names.isdisjoint(simple_names)
