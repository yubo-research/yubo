import pytest


class _MockPolicy:
    def __init__(self, num_params=5):
        self._num_params = num_params

    def num_params(self):
        return self._num_params


def test_metric_turbo_enn_designer_forces_python_backend(monkeypatch):
    from optimizer import turbo_enn_designer as ted
    from optimizer import turbo_enn_designer_ext as ted_ext

    calls = []

    def fake_py(*, bounds, config, rng):
        calls.append(("py", bounds.shape))
        return object()

    def fake_auto(*, bounds, config, rng):
        calls.append(("auto", bounds.shape))
        return object()

    monkeypatch.setattr(ted_ext, "_create_optimizer_py_local", fake_py)
    monkeypatch.setattr(ted, "_create_optimizer_auto", fake_auto)

    designer = ted_ext.TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="thompson",
        tr_geometry="enn_metr",
        covmat="low_rank",
        metric_rank=10,
    )
    monkeypatch.setattr(designer, "_make_config", lambda num_init, num_metrics: object())
    monkeypatch.setattr(designer, "_resolve_num_metrics", lambda data: None)
    designer._num_arms = 1
    designer._init_optimizer([], 1)

    assert calls == [("py", (7, 2))]


def test_metric_turbo_enn_designer_builds_local_metric_tr_state():
    from optimizer.metric_trust_region import ENNMetricShapedTrustRegion
    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="pareto",
        tr_geometry="enn_metr",
        covmat="low_rank",
        metric_rank=3,
    )
    designer._num_arms = 1
    designer._init_optimizer([], 1)

    assert type(designer._turbo).__name__ == "TurboOptimizer"
    assert isinstance(designer._turbo._tr_state, ENNMetricShapedTrustRegion)


def test_metric_turbo_enn_designer_enables_accel_tr_in_config():
    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="pareto",
        tr_geometry="enn_metr",
        covmat="dense",
    )

    tr_cfg = designer._make_trust_region(None)

    assert tr_cfg.use_accel is True


def test_ellipsoid_turbo_enn_designer_builds_local_ellipsoid_tr_state():
    from optimizer.ellipsoidal_trust_region import ENNTrueEllipsoidalTrustRegion
    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="pareto",
        tr_geometry="enn_ellip",
        covmat="low_rank",
        update_option="option_c",
    )
    designer._num_arms = 1
    designer._init_optimizer([], 1)

    assert type(designer._turbo).__name__ == "TurboOptimizer"
    assert isinstance(designer._turbo._tr_state, ENNTrueEllipsoidalTrustRegion)


def test_box_turbo_enn_designer_keeps_auto_backend(monkeypatch):
    from optimizer import turbo_enn_designer as ted

    calls = []

    def fake_py(*, bounds, config, rng):
        calls.append(("py", bounds.shape))
        return object()

    def fake_auto(*, bounds, config, rng):
        calls.append(("auto", bounds.shape))
        return object()

    monkeypatch.setattr(ted, "_create_optimizer_py", fake_py)
    monkeypatch.setattr(ted, "_create_optimizer_auto", fake_auto)

    designer = ted.TurboENNDesigner(
        _MockPolicy(num_params=5),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="thompson",
    )
    monkeypatch.setattr(designer, "_make_config", lambda num_init, num_metrics: object())
    monkeypatch.setattr(designer, "_resolve_num_metrics", lambda data: None)
    designer._num_arms = 1
    designer._init_optimizer([], 1)

    assert calls == [("auto", (5, 2))]


def test_turbo_enn_fit_accepts_optional_k(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(ctx, {"acq_type": "ucb", "k": 16, "candidate_rv": "sobol"})
    assert out["config"].k == 16
    assert out["config"].candidate_rv == "sobol"
    assert calls[0]["config"].k == 16
    assert calls[0]["config"].candidate_rv == "sobol"


def test_turbo_enn_fit_forwards_fit_and_fixed_length_options(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        10,
        None,
        num_keep=8000,
        keep_style="trailing",
        num_keep_val=8000,
        init_yubo_default=10,
        init_ax_default=10,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(
        ctx,
        {
            "acq_type": "thompson",
            "candidate_rv": "sobol",
            "num_candidates": 64,
            "num_fit_samples": 10,
            "num_fit_candidates": 100,
            "geometry": "enn_metr",
            "covmat": "dense",
            "fixed_length": 1.6,
        },
    )

    cfg = out["config"]
    assert cfg.num_candidates == 64
    assert cfg.num_fit_samples == 10
    assert cfg.num_fit_candidates == 100
    assert cfg.trust_region.fixed_length == pytest.approx(1.6)
    assert calls[0]["config"].num_keep == 8000


def test_turbo_enn_fit_forwards_ellipsoid_option_surface(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(
        ctx,
        {
            "acq_type": "thompson",
            "geometry": "enn_ellip",
            "covmat": "low_rank",
            "rank": 10,
            "p_raasp": 0.3,
            "radial_mode": "boundary",
            "shape_period": 7,
            "shape_ema": 0.4,
            "shape_jitter": 1e-5,
            "shape_kappa_max": 5e3,
            "rho_bad": 0.2,
            "rho_good": 0.8,
            "gamma_down": 0.4,
            "gamma_up": 2.5,
            "boundary_tol": 0.05,
        },
    )

    cfg = out["config"]
    tr = cfg.trust_region
    assert tr.p_raasp == pytest.approx(0.3)
    assert tr.radial_mode == "boundary"
    assert tr.shape_period == 7
    assert tr.shape_ema == pytest.approx(0.4)
    assert tr.shape_jitter == pytest.approx(1e-5)
    assert tr.shape_kappa_max == pytest.approx(5e3)
    assert tr.rho_bad == pytest.approx(0.2)
    assert tr.rho_good == pytest.approx(0.8)
    assert tr.gamma_down == pytest.approx(0.4)
    assert tr.gamma_up == pytest.approx(2.5)
    assert tr.boundary_tol == pytest.approx(0.05)
    assert calls[0]["config"].trust_region.shape_jitter == pytest.approx(1e-5)
    assert calls[0]["config"].trust_region.shape_kappa_max == pytest.approx(5e3)


def test_turbo_enn_designer_ext_accepts_multi_region_knobs():
    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="ucb",
        tr_geometry="enn_ellip",
        covmat="low_rank",
        metric_rank=3,
        tr_length_fixed=1.6,
        p_raasp=0.3,
        radial_mode="boundary",
        shape_jitter=1e-5,
        shape_kappa_max=5e3,
    )

    assert designer._tr_length_fixed == pytest.approx(1.6)
    assert designer._p_raasp == pytest.approx(0.3)
    assert designer._radial_mode == "boundary"
    assert designer._shape_jitter == pytest.approx(1e-5)
    assert designer._shape_kappa_max == pytest.approx(5e3)


def test_turbo_enn_multi_forwards_region_rng_and_richer_knobs(monkeypatch):
    import numpy as np

    from optimizer import multi_turbo_enn_designer as mtd
    from optimizer.turbo_enn_designer_ext import TurboENNTrustRegionSpec

    captured: list[tuple[np.random.Generator | None, dict]] = []

    class _Child:
        def __init__(self, _policy, *, config=None, **kwargs):
            captured.append((config, kwargs))

        def best_datum(self):
            return None

        def state_dict(self, data=None):
            _ = data
            return {}

    monkeypatch.setattr(mtd, "TurboENNDesigner", _Child)

    designer = mtd.MultiTurboENNDesigner(
        _MockPolicy(num_params=7),
        config=mtd.MultiTurboENNConfig(
            harness=mtd.MultiTurboHarnessConfig(num_regions=2, strategy="shared_data", arm_mode="allocated"),
            region=mtd.TurboENNRegionConfig(
                turbo_mode="turbo-enn",
                acq_type="ucb",
                trust_region=TurboENNTrustRegionSpec(
                    tr_type="turbo",
                    tr_geometry="box",
                    covmat=None,
                    metric_rank=None,
                    fixed_length=1.6,
                    update_option="option_a",
                    p_raasp=0.3,
                    radial_mode="boundary",
                    shape_period=5,
                    shape_ema=0.2,
                    shape_jitter=1e-6,
                    shape_kappa_max=1e4,
                    rho_bad=0.25,
                    rho_good=0.75,
                    gamma_down=0.5,
                    gamma_up=2.0,
                    boundary_tol=0.1,
                    use_accel=False,
                    accel=None,
                ),
            ),
        ),
        rng=np.random.default_rng(0),
    )
    designer._init_regions([], 1)

    assert len(captured) == 2
    for config, kwargs in captured:
        assert isinstance(config.rng, np.random.Generator)
        assert config.trust_region.fixed_length == pytest.approx(1.6)
        assert config.trust_region.p_raasp == pytest.approx(0.3)
        assert config.trust_region.radial_mode == "boundary"
        assert kwargs == {}


def test_turbo_enn_fit_rejects_unknown_option():
    from optimizer import designer_registry as dr
    from optimizer.designer_errors import NoSuchDesignerError

    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    with pytest.raises(NoSuchDesignerError, match="does not support options"):
        dr._d_turbo_enn_fit(ctx, {"acq_type": "ucb", "unknown_knob": 64})


def test_turbo_enn_fit_rejects_explicit_box_geometry():
    from optimizer import designer_registry as dr
    from optimizer.designer_errors import NoSuchDesignerError

    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    with pytest.raises(NoSuchDesignerError, match="must be one of"):
        dr._d_turbo_enn_fit(ctx, {"acq_type": "ucb", "geometry": "box"})


def test_turbo_enn_fit_plain_does_not_inject_metric_options(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(ctx, {"acq_type": "thompson"})
    cfg = out["config"]
    assert cfg.trust_region.tr_geometry == "box"
    assert cfg.trust_region.covmat is None
    assert cfg.trust_region.metric_rank is None
    assert calls[0]["config"].trust_region.covmat is None
    assert calls[0]["config"].trust_region.metric_rank is None


def test_turbo_enn_fit_accepts_identity_geometry(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(ctx, {"acq_type": "ucb", "geometry": "enn_iso"})
    assert out["config"].trust_region.tr_geometry == "enn_iso"
    assert calls[0]["config"].trust_region.tr_geometry == "enn_iso"


def test_turbo_enn_fit_non_box_defaults_to_accel(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(ctx, {"acq_type": "ucb", "geometry": "enn_ellip"})
    assert out["config"].trust_region.use_accel is True
    assert out["config"].trust_region.accel is None
    assert calls[0]["config"].trust_region.use_accel is True


def test_turbo_enn_fit_forwards_explicit_accel_override(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(
        ctx,
        {"acq_type": "thompson", "geometry": "enn_ellip", "use_accel": True, "accel": "triton"},
    )
    assert out["config"].trust_region.use_accel is True
    assert out["config"].trust_region.accel == "triton"
    assert calls[0]["config"].trust_region.accel == "triton"


def test_turbo_enn_fit_rejects_gradient_identity_geometry(monkeypatch):
    from optimizer import designer_registry as dr
    from optimizer.designer_errors import NoSuchDesignerError

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    with pytest.raises(NoSuchDesignerError, match="must be one of"):
        dr._d_turbo_enn_fit(ctx, {"acq_type": "ucb", "geometry": "grad_iso"})


def test_metric_turbo_enn_designer_builds_fixed_length_tr_config():
    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="thompson",
        tr_geometry="enn_metr",
        covmat="dense",
        fixed_length=1.6,
    )

    tr_cfg = designer._make_trust_region(None)

    assert tr_cfg.fixed_length == pytest.approx(1.6)


def test_metric_turbo_enn_designer_keeps_fixed_length_on_box_fast_path():
    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="ucb",
        tr_geometry="box",
        fixed_length=1.6,
    )

    tr_cfg = designer._make_trust_region(None)

    assert tr_cfg.fixed_length == pytest.approx(1.6)


def test_metric_turbo_enn_designer_box_fast_path_stays_plain_without_fixed_length():
    from enn.turbo.config.trust_region import TurboTRConfig

    from optimizer.turbo_enn_designer_ext import TurboENNDesigner

    designer = TurboENNDesigner(
        _MockPolicy(num_params=7),
        turbo_mode="turbo-enn",
        k=10,
        acq_type="ucb",
        tr_geometry="box",
    )

    tr_cfg = designer._make_trust_region(None)

    assert isinstance(tr_cfg, TurboTRConfig)


def test_turbo_enn_p_metric_forwards_geometry_and_disables_fit(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_p(
        ctx,
        {
            "geometry": "enn_metr",
            "covmat": "low_rank",
            "rank": 10,
        },
    )
    cfg = out["config"]
    assert cfg.acq_type == "pareto"
    assert cfg.num_fit_samples is None
    assert cfg.num_fit_candidates is None
    assert cfg.trust_region.tr_geometry == "enn_metr"
    assert cfg.trust_region.covmat == "low_rank"
    assert cfg.trust_region.metric_rank == 10
    assert calls[0]["config"].num_fit_samples is None
    assert calls[0]["config"].num_fit_candidates is None


def test_turbo_enn_p_ellipsoid_forwards_update_option(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_p(
        ctx,
        {
            "geometry": "enn_ellip",
            "covmat": "low_rank",
            "update_option": "option_c",
        },
    )
    cfg = out["config"]
    assert cfg.trust_region.tr_geometry == "enn_ellip"
    assert cfg.trust_region.covmat == "low_rank"
    assert cfg.trust_region.update_option == "option_c"
    assert calls[0]["config"].trust_region.update_option == "option_c"


def test_turbo_enn_multi_ext_forwards_multi_region_options(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn_multi(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn_multi", fake_turbo_enn_multi)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_multi_ext(
        ctx,
        {
            "acq_type": "ucb",
            "num_regions": 4,
            "strategy": "shared_data",
            "arm_mode": "allocated",
            "pool_multiplier": 3,
        },
    )
    assert out["num_regions"] == 4
    assert out["strategy"] == "shared_data"
    assert out["arm_mode"] == "allocated"
    assert out["pool_multiplier"] == 3
    assert out["tr_geometry"] == "box"
    assert calls[0]["turbo_mode"] == "turbo-enn"


def test_turbo_py_enn_p_forces_python(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_py_enn_p(ctx, {"geometry": "grad_metr"})
    assert out["config"].trust_region.tr_geometry == "grad_metr"
    assert out["config"].use_python is True
    assert calls[0]["config"].use_python is True


def test_turbo_enn_fit_metric_forwards_rank(monkeypatch):
    from optimizer import designer_registry as dr

    calls = []

    def fake_turbo_enn(ctx, **kw):
        calls.append(kw)
        return kw

    monkeypatch.setattr(dr, "_turbo_enn", fake_turbo_enn)
    monkeypatch.setattr(dr, "_turbo_enn_ext", fake_turbo_enn)
    ctx = dr._SimpleContext(
        _MockPolicy(),
        1,
        None,
        num_keep=None,
        keep_style="trailing",
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=64,
        env_conf=None,
    )

    out = dr._d_turbo_enn_fit(
        ctx,
        {
            "acq_type": "thompson",
            "geometry": "enn_metr",
            "covmat": "low_rank",
            "rank": 10,
        },
    )
    cfg = out["config"]
    assert cfg.trust_region.tr_geometry == "enn_metr"
    assert cfg.trust_region.covmat == "low_rank"
    assert cfg.trust_region.metric_rank == 10
    assert calls[0]["config"].trust_region.metric_rank == 10
