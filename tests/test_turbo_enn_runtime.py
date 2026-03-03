from types import SimpleNamespace

import numpy as np
import pytest

from optimizer import turbo_enn_runtime as runtime


def _make_datum(*, params, rreturn, rreturn_se=None):
    policy = SimpleNamespace(get_params=lambda: np.asarray(params, dtype=float))
    trajectory = SimpleNamespace(rreturn=rreturn, rreturn_se=rreturn_se, rreturn_est=None)
    return SimpleNamespace(policy=policy, trajectory=trajectory)


def test_resolve_num_metrics_non_morbo_returns_existing_value():
    designer = SimpleNamespace(_num_metrics=7, _tr_type="turbo", _policy=SimpleNamespace())
    assert runtime.resolve_num_metrics(designer, data=[]) == 7


def test_resolve_num_metrics_morbo_infers_from_policy_and_data():
    designer = SimpleNamespace(
        _num_metrics=None,
        _tr_type="morbo",
        _policy=SimpleNamespace(num_metrics=lambda: 3),
    )
    assert runtime.resolve_num_metrics(designer, data=[]) == 3
    assert designer._num_metrics == 3

    designer_data = SimpleNamespace(
        _num_metrics=None,
        _tr_type="morbo",
        _policy=SimpleNamespace(),
    )
    data = [_make_datum(params=[0.0, 1.0], rreturn=np.array([1.0, 2.0], dtype=float))]
    assert runtime.resolve_num_metrics(designer_data, data=data) == 2


def test_resolve_num_metrics_morbo_requires_at_least_two():
    designer = SimpleNamespace(
        _num_metrics=None,
        _tr_type="morbo",
        _policy=SimpleNamespace(num_metrics=lambda: 1),
    )
    with pytest.raises(ValueError, match="num_metrics must be >= 2"):
        _ = runtime.resolve_num_metrics(designer, data=[])


def test_tell_new_data_updates_estimate_and_passes_yvar_when_enabled():
    class _Turbo:
        def __init__(self):
            self.calls = []

        def tell(self, x, y_obs, y_var=None):
            self.calls.append((np.array(x), np.array(y_obs), None if y_var is None else np.array(y_var)))
            return np.array(y_obs, dtype=float) + 0.5

    turbo = _Turbo()
    designer = SimpleNamespace(
        _tr_type="turbo",
        _use_y_var=False,
        _turbo=turbo,
        _datum_best=None,
        _y_est_best=None,
    )
    new_data = [
        _make_datum(params=[0.0, 0.0], rreturn=1.0),
        _make_datum(params=[1.0, 1.0], rreturn=3.0),
    ]
    runtime.tell_new_data(designer, new_data)
    assert new_data[0].trajectory.rreturn_est == pytest.approx(1.5)
    assert new_data[1].trajectory.rreturn_est == pytest.approx(3.5)
    assert designer._datum_best is new_data[1]
    assert designer._y_est_best == pytest.approx(3.5)

    turbo_yvar = _Turbo()
    designer_yvar = SimpleNamespace(
        _tr_type="morbo",
        _use_y_var=True,
        _turbo=turbo_yvar,
        _datum_best=None,
        _y_est_best=None,
    )
    yvar_data = [
        _make_datum(params=[0.0], rreturn=1.0, rreturn_se=0.2),
        _make_datum(params=[1.0], rreturn=2.0, rreturn_se=0.3),
    ]
    runtime.tell_new_data(designer_yvar, yvar_data)
    assert turbo_yvar.calls and turbo_yvar.calls[0][2] is not None
    np.testing.assert_allclose(turbo_yvar.calls[0][2].reshape(-1), np.array([0.04, 0.09], dtype=float))


def test_update_best_estimate_respects_existing_best():
    sentinel = object()
    designer = SimpleNamespace(_y_est_best=10.0, _datum_best=sentinel)
    data = [_make_datum(params=[0.0], rreturn=1.0), _make_datum(params=[1.0], rreturn=2.0)]
    runtime.update_best_estimate(designer, data, np.array([3.0, 5.0], dtype=float))
    assert designer._datum_best is sentinel
    assert designer._y_est_best == pytest.approx(10.0)


def test_call_designer_initializes_tells_and_emits_policies_with_telemetry():
    events = []

    class _Turbo:
        def ask(self, num_arms):
            return np.arange(num_arms * 2, dtype=float).reshape(num_arms, 2)

        def telemetry(self):
            return SimpleNamespace(dt_fit=0.25, dt_sel=0.5)

    class _Telemetry:
        def __init__(self):
            self.fit = None
            self.select = None

        def set_dt_fit(self, value):
            self.fit = float(value)

        def set_dt_select(self, value):
            self.select = float(value)

    designer = SimpleNamespace(_num_arms=None, _num_told=0, _turbo=None)

    def _init_optimizer(data, num_arms):
        events.append(("init", len(data), int(num_arms)))
        designer._turbo = _Turbo()

    def _tell_new_data(new_data):
        events.append(("tell", len(new_data)))

    designer._init_optimizer = _init_optimizer
    designer._tell_new_data = _tell_new_data
    designer._make_policy = lambda x: tuple(np.asarray(x, dtype=float).tolist())

    data = [object(), object(), object()]
    telemetry = _Telemetry()
    policies = runtime.call_designer(designer, data, num_arms=2, telemetry=telemetry)
    assert policies == [(0.0, 1.0), (2.0, 3.0)]
    assert designer._num_arms == 2
    assert designer._num_told == 3
    assert events == [("init", 3, 2), ("tell", 3)]
    assert telemetry.fit == pytest.approx(0.25)
    assert telemetry.select == pytest.approx(0.5)

    policies_again = runtime.call_designer(designer, data, num_arms=2, telemetry=None)
    assert policies_again == [(0.0, 1.0), (2.0, 3.0)]
    assert events == [("init", 3, 2), ("tell", 3)]


def test_get_algo_metrics_collects_supported_fields():
    empty = runtime.get_algo_metrics(SimpleNamespace(_turbo=None))
    assert empty == {}

    turbo = SimpleNamespace(
        tr_length=1.5,
        tr_obs_count=7,
        telemetry=lambda: SimpleNamespace(dt_fit=0.3, dt_sel=0.7),
    )
    out = runtime.get_algo_metrics(SimpleNamespace(_turbo=turbo))
    assert out["tr_length"] == pytest.approx(1.5)
    assert out["tr_obs"] == pytest.approx(7.0)
    assert out["fit_dt"] == pytest.approx(0.3)
    assert out["select_dt"] == pytest.approx(0.7)
