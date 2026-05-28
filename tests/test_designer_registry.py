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


def test_turbo_enn_p_accepts_index_driver_option():
    from optimizer.designer_registry import _DESIGNER_OPTION_SPECS, _d_turbo_enn_p

    ctx = _make_ctx()
    d = _d_turbo_enn_p(ctx, {"idx": "hnsw"})

    assert d._index_driver == "hnsw"
    assert any(spec.name == "idx" for spec in _DESIGNER_OPTION_SPECS["turbo-enn-p"])


def test_turbo_enn_fit_ucb_accepts_exact_index_alias():
    from optimizer.designer_registry import _d_turbo_enn_fit_ucb

    ctx = _make_ctx()
    d = _d_turbo_enn_fit_ucb(ctx, {"idx": "exact"})

    assert d._index_driver == "flat"


def test_d_turbo_enn_fit_ucb_rejects_unknown_option():
    from optimizer.designer_registry import _d_turbo_enn_fit_ucb

    ctx = _make_ctx()
    with pytest.raises(NoSuchDesignerError, match="does not support"):
        _d_turbo_enn_fit_ucb(ctx, {"bogus": 1})


def test_d_turbo_enn_p_defaults_flat():
    from optimizer.designer_registry import _d_turbo_enn_p

    ctx = _make_ctx()
    d = _d_turbo_enn_p(ctx, {})
    assert d._index_driver is None


def test_d_turbo_enn_p_idx_hnsw():
    from optimizer.designer_registry import _d_turbo_enn_p

    ctx = _make_ctx()
    d = _d_turbo_enn_p(ctx, {"idx": "hnsw"})
    assert d._index_driver == "hnsw"


def test_d_turbo_enn_p_rejects_unknown_option():
    from optimizer.designer_registry import _d_turbo_enn_p

    ctx = _make_ctx()
    with pytest.raises(NoSuchDesignerError, match="does not support"):
        _d_turbo_enn_p(ctx, {"k": 3})


def test_d_turbo_enn_fit_ucb_idx_hnsw():
    from optimizer.designer_registry import _d_turbo_enn_fit_ucb

    ctx = _make_ctx()
    d = _d_turbo_enn_fit_ucb(ctx, {"idx": "hnsw", "k": 7})
    assert d._index_driver == "hnsw"
    assert d._k == 7


def test_turbo_enn_p_hnsw_alias_builds():
    from optimizer.designer_registry import _DESIGNER_DISPATCH

    ctx = _make_ctx()
    d = _DESIGNER_DISPATCH["turbo-enn-p-hnsw"](ctx, {})
    assert d._index_driver == "hnsw"


def test_build_turbo_enn_turbo_enn_p_kind():
    from optimizer.designer_registry import _build_turbo_enn

    ctx = _make_ctx()
    d = _build_turbo_enn(ctx, "turbo-enn-p")
    assert d is not None
    assert d._turbo_mode == "turbo-enn"


def test_turbo_make_config_index_driver_hnsw():
    from enn.turbo.config.enn_index_driver import ENNIndexDriver

    from optimizer.turbo_enn_designer import TurboENNDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)
    designer = TurboENNDesigner(policy, turbo_mode="turbo-enn", k=3, index_driver="hnsw")
    cfg = designer._make_config(num_init=5, num_metrics=None)
    assert cfg.surrogate.index_driver is ENNIndexDriver.HNSW


@pytest.mark.parametrize("turbo_mode", ["turbo-one", "turbo-zero", "turbo-enn", "lhd-only"])
def test_turbo_make_config_matches_enn_factory(turbo_mode):
    """Regression: enn factory configs no longer accept trailing_obs."""
    from optimizer.turbo_enn_designer import TurboENNDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)
    kwargs = {"turbo_mode": turbo_mode, "num_init": 5}
    if turbo_mode == "turbo-enn":
        kwargs["k"] = 3
    designer = TurboENNDesigner(policy, **kwargs)
    designer._make_config(num_init=5, num_metrics=None)


def test_turbo_one_and_turbo_one_nds_have_different_acq_types():
    """Regression: turbo-one should use thompson, turbo-one-nds should use pareto.

    These were originally different acquisition types. Commit 13fd7aae accidentally
    changed turbo-one from "thompson" to "pareto", making them identical.
    """
    from optimizer.designer_registry import _build_turbo_enn

    ctx = _make_ctx()
    turbo_one = _build_turbo_enn(ctx, "turbo-one")
    turbo_one_nds = _build_turbo_enn(ctx, "turbo-one-nds")

    assert turbo_one._acq_type == "thompson", "turbo-one should use thompson acquisition"
    assert turbo_one_nds._acq_type == "pareto", "turbo-one-nds should use pareto acquisition"
    assert turbo_one._acq_type != turbo_one_nds._acq_type, "turbo-one and turbo-one-nds should differ"
