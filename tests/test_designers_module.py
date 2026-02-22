import numpy as np
import pytest


class MockPolicy:
    def __init__(self, num_params=5):
        self._num_params = num_params
        self.problem_seed = 0

    def num_params(self):
        return self._num_params

    def set_params(self, x):
        pass

    def get_params(self):
        return np.zeros(self._num_params)

    def clone(self):
        return MockPolicy(self._num_params)


def test_designers_init():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    assert designers is not None


def test_designers_create_random():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("random")
    assert designer is not None


def test_designers_create_sobol():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("sobol")
    assert designer is not None


def test_designers_create_lhd():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("lhd")
    assert designer is not None


def test_designers_create_center():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("center")
    assert designer is not None


def test_designers_no_such_designer():
    from optimizer.designers import Designers, NoSuchDesignerError

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    with pytest.raises(NoSuchDesignerError):
        designers.create("nonexistent_designer")


def test_designers_is_valid():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    assert designers.is_valid("random") is True
    assert designers.is_valid("nonexistent_designer") is False


def test_designers_catalog():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    catalog = designers.catalog()
    assert isinstance(catalog, list)
    names = {entry.base_name for entry in catalog}
    assert "sobol" in names


def test_designers_catalog_dataclasses_are_instantiable():
    # This is intentionally direct: it ensures the catalog-related dataclasses
    # are covered by tests (for `kiss check` coverage gating).
    from optimizer.designers import (
        DesignerCatalogEntry,
        DesignerOptionSpec,
        DesignerSpec,
    )

    opt = DesignerOptionSpec(
        name="k",
        required=True,
        value_type="int",
        description="dummy",
        example="x/k=1",
        allowed_values=None,
    )
    entry = DesignerCatalogEntry(base_name="x", options=[opt], dispatch=lambda *_: None)
    spec = DesignerSpec(base="x", general={}, specific={})

    assert entry.base_name == "x"
    assert entry.options[0].name == "k"
    assert spec.base == "x"


def _mk_ctx(policy):
    from optimizer.designer_registry import _SimpleContext

    return _SimpleContext(
        policy=policy,
        num_arms=4,
        bt=lambda *args, **kwargs: (args, kwargs),
        num_keep=None,
        keep_style=None,
        num_keep_val=None,
        init_yubo_default=4,
        init_ax_default=8,
        default_num_X_samples=64,
    )


def test_turbo_enn_fit_ext_parses_optional_options(monkeypatch):
    from optimizer import designer_registry as dr

    captured = {}

    def _fake(ctx, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dr, "_turbo_enn_ext", _fake)
    _ = dr._d_turbo_enn_fit_ext_v2(
        _mk_ctx(MockPolicy()),
        {
            "acq_type": "ucb",
            "num_candidates": 64,
            "num_fit_samples": 32,
            "num_fit_candidates": 48,
            "geometry": "enn_metric_shaped",
            "sampler": "low_rank",
            "rank": 5,
            "tr_length_fixed": 1.6,
            "candidate_rv": "uniform",
        },
    )
    assert captured["acq_type"] == "ucb"
    assert captured["num_candidates"] == 64
    assert captured["tr_geometry"] == "enn_metric_shaped"
    assert captured["metric_sampler"] == "low_rank"
    assert captured["metric_rank"] == 5
    assert captured["tr_length_fixed"] == 1.6
    assert captured["candidate_rv"] == "uniform"


def test_turbo_enn_multi_ext_requires_num_regions(monkeypatch):
    from optimizer import designer_registry as dr
    from optimizer.designers import NoSuchDesignerError

    monkeypatch.setattr(dr, "_turbo_enn_multi", lambda ctx, **kwargs: kwargs)
    with pytest.raises(NoSuchDesignerError):
        dr._d_turbo_enn_multi_ext(
            _mk_ctx(MockPolicy()),
            {"acq_type": "ucb"},
        )


def test_turbo_enn_fit_true_ellipsoid_options(monkeypatch):
    from optimizer import designer_registry as dr

    captured = {}

    def _fake(ctx, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dr, "_turbo_enn_ext", _fake)
    _ = dr._d_turbo_enn_fit_ext_v2(
        _mk_ctx(MockPolicy()),
        {
            "acq_type": "ucb",
            "geometry": "enn_true_ellipsoid",
            "update_option": "option_c",
            "p_raasp": 0.4,
            "radial_mode": "boundary",
            "rho_bad": 0.2,
            "rho_good": 0.8,
            "gamma_down": 0.6,
            "gamma_up": 1.5,
            "boundary_tol": 0.05,
        },
    )
    assert captured["tr_geometry"] == "enn_true_ellipsoid"
    assert captured["update_option"] == "option_c"
    assert captured["p_raasp"] == 0.4
    assert captured["radial_mode"] == "boundary"
    assert captured["rho_bad"] == 0.2
    assert captured["rho_good"] == 0.8
    assert captured["gamma_down"] == 0.6
    assert captured["gamma_up"] == 1.5
    assert captured["boundary_tol"] == 0.05


def test_designers_is_valid_for_multi_turbo_names():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=2)
    assert designers.is_valid("turbo-enn-multi/acq_type=ucb/num_regions=2") is True
    assert designers.is_valid("morbo-enn-multi/acq_type=pareto/num_regions=3/strategy=shared_data") is True


def test_designers_create_multi_turbo():
    from optimizer.designers import Designers
    from optimizer.multi_turbo_enn_designer import MultiTurboENNDesigner

    policy = MockPolicy()
    designers = Designers(policy, num_arms=2)
    designer = designers.create("turbo-enn-multi/acq_type=ucb/num_regions=2")
    assert isinstance(designer, MultiTurboENNDesigner)


def test_cma_designer_get_algo_metrics():
    from optimizer.cma_designer import CMAESDesigner

    policy = MockPolicy(num_params=3)
    designer = CMAESDesigner(policy)
    assert designer.get_algo_metrics() == {}

    policies = designer([], num_arms=4)
    assert len(policies) == 4
    metrics = designer.get_algo_metrics()
    assert "sigma" in metrics
    assert isinstance(metrics["sigma"], (int, float))
