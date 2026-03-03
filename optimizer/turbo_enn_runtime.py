from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.designer_asserts import assert_scalar_rreturn


def resolve_num_metrics(designer: Any, data: list) -> int | None:
    num_metrics = designer._num_metrics
    if designer._tr_type != "morbo":
        return num_metrics
    if num_metrics is None:
        num_metrics = infer_num_metrics(designer, data)
    if num_metrics < 2:
        raise ValueError("num_metrics must be >= 2 for tr_type='morbo'")
    designer._num_metrics = num_metrics
    return num_metrics


def infer_num_metrics(designer: Any, data: list) -> int:
    policy_metrics = getattr(designer._policy, "num_metrics", None)
    if callable(policy_metrics):
        policy_metrics = policy_metrics()
    if policy_metrics is not None:
        return int(policy_metrics)
    if len(data) > 0:
        y = np.asarray([d.trajectory.rreturn for d in data])
        return int(y.shape[1]) if y.ndim == 2 else 1
    return 2


def tell_new_data(designer: Any, new_data: list) -> None:
    if designer._tr_type != "morbo":
        assert_scalar_rreturn(new_data)
    x_list = [d.policy.get_params() for d in new_data]
    y_list = [d.trajectory.rreturn for d in new_data]
    y_se_list = [d.trajectory.rreturn_se for d in new_data] if designer._use_y_var else []
    if designer._use_y_var:
        assert all(se is not None for se in y_se_list)
    if len(x_list) == 0:
        return
    x = np.array(x_list)
    y_obs = np.array(y_list)
    y_obs = y_obs[:, None] if y_obs.ndim == 1 else y_obs
    y_est = designer._turbo.tell(x, y_obs, y_var=np.array(y_se_list) ** 2) if y_se_list else designer._turbo.tell(x, y_obs)
    assert y_obs.shape == y_est.shape and y_obs.shape[0] == len(new_data)
    if y_est.shape[1] == 1:
        update_best_estimate(designer, new_data, y_est[:, 0])


def update_best_estimate(designer: Any, new_data: list, y_est_0: np.ndarray) -> None:
    y_est_0 = np.asarray(y_est_0, dtype=np.float64)
    for i, datum in enumerate(new_data):
        datum.trajectory.rreturn_est = float(y_est_0[i])
    best_i = int(np.argmax(y_est_0))
    best_y = float(y_est_0[best_i])
    if designer._y_est_best is None or best_y > float(designer._y_est_best):
        designer._y_est_best = best_y
        designer._datum_best = new_data[best_i]


def call_designer(designer: Any, data: list, num_arms: int, *, telemetry: Any = None) -> list:
    if designer._num_arms is None:
        designer._num_arms = num_arms
        designer._init_optimizer(data, num_arms)

    if len(data) > designer._num_told:
        designer._tell_new_data(data[designer._num_told :])
        designer._num_told = len(data)

    x_new = designer._turbo.ask(num_arms)
    if telemetry is not None:
        tel = designer._turbo.telemetry()
        telemetry.set_dt_fit(tel.dt_fit)
        telemetry.set_dt_select(tel.dt_sel)
    return [designer._make_policy(x) for x in x_new]


def get_algo_metrics(designer: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    turbo = getattr(designer, "_turbo", None)
    if turbo is None:
        return out
    if hasattr(turbo, "tr_length"):
        try:
            out["tr_length"] = float(turbo.tr_length)
        except (TypeError, ValueError):
            pass
    if hasattr(turbo, "tr_obs_count"):
        try:
            out["tr_obs"] = float(turbo.tr_obs_count)
        except (TypeError, ValueError):
            pass
    tel_fn = getattr(turbo, "telemetry", None)
    if callable(tel_fn):
        tel = tel_fn()
        if tel is not None:
            if hasattr(tel, "dt_fit") and tel.dt_fit is not None:
                out["fit_dt"] = float(tel.dt_fit)
            if hasattr(tel, "dt_sel") and tel.dt_sel is not None:
                out["select_dt"] = float(tel.dt_sel)
    return out
