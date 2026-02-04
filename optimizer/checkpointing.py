from __future__ import annotations

import json
import pickle
from typing import Any, NamedTuple

import numpy as np
import torch

from .datum import Datum
from .trajectories import Trajectory


class _TraceEntry(NamedTuple):
    rreturn: float
    rreturn_decision: float
    dt_prop: float
    dt_eval: float


class _DesignerSnapshot(NamedTuple):
    state: np.ndarray | None
    class_name: str | None
    index: int | None


def _pack_pickle(obj: Any) -> np.ndarray:
    return np.frombuffer(pickle.dumps(obj), dtype=np.uint8)


def _unpack_pickle(arr: np.ndarray) -> object:
    return pickle.loads(arr.tobytes())


def _policy_init_flat(policy_template: Any) -> np.ndarray | None:
    value = getattr(policy_template, "_flat_params_init", None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _set_policy_baseline(policy: Any, baseline: np.ndarray | None) -> None:
    if baseline is None or not hasattr(policy, "_flat_params_init"):
        return
    flat = np.asarray(baseline, dtype=np.float32)
    policy._flat_params_init = flat.copy()
    idx = 0
    for param in policy.parameters():
        size = param.numel()
        shape = param.shape
        param.data.copy_(torch.as_tensor(flat[idx : idx + size].reshape(shape)))
        idx += size


def _serialize_observations(opt: Any) -> dict[str, np.ndarray]:
    params, rreturn, rreturn_se, rreturn_est = [], [], [], []
    for datum in opt._data:
        params.append(np.asarray(datum.policy.get_params(), dtype=np.float32))
        rr = np.asarray(datum.trajectory.rreturn, dtype=np.float64)
        rreturn.append(rr)
        rreturn_se.append(np.full_like(rr, np.nan) if datum.trajectory.rreturn_se is None else np.asarray(datum.trajectory.rreturn_se, dtype=np.float64))
        rreturn_est.append(np.full_like(rr, np.nan) if datum.trajectory.rreturn_est is None else np.asarray(datum.trajectory.rreturn_est, dtype=np.float64))

    params_arr = np.stack(params) if params else np.zeros((0, int(opt.num_params)), dtype=np.float32)
    rreturn_arr = np.stack(rreturn) if rreturn else np.zeros((0,), dtype=np.float64)
    rreturn_se_arr = np.stack(rreturn_se) if rreturn_se else np.zeros_like(rreturn_arr)
    rreturn_est_arr = np.stack(rreturn_est) if rreturn_est else np.zeros_like(rreturn_arr)
    return {
        "params": params_arr,
        "rreturn": rreturn_arr,
        "rreturn_se": rreturn_se_arr,
        "rreturn_est": rreturn_est_arr,
    }


def _serialize_trace(opt: Any) -> dict[str, np.ndarray]:
    trace_rreturn = np.asarray([t.rreturn for t in opt._trace], dtype=np.float64)
    trace_rreturn_decision = np.asarray([t.rreturn_decision for t in opt._trace], dtype=np.float64)
    trace_dt_prop = np.asarray([t.dt_prop for t in opt._trace], dtype=np.float64)
    trace_dt_eval = np.asarray([t.dt_eval for t in opt._trace], dtype=np.float64)
    return {
        "trace_rreturn": trace_rreturn,
        "trace_rreturn_decision": trace_rreturn_decision,
        "trace_dt_prop": trace_dt_prop,
        "trace_dt_eval": trace_dt_eval,
    }


def _serialize_best(opt: Any) -> tuple[np.ndarray | None, int | None]:
    best_params = None
    if opt.best_policy is not None:
        best_params = np.asarray(opt.best_policy.get_params(), dtype=np.float32)

    best_index = None
    if opt.best_datum is not None:
        try:
            best_index = int(opt._data.index(opt.best_datum))
        except ValueError:
            best_index = None
    return best_params, best_index


def _serialize_designer(opt: Any) -> _DesignerSnapshot:
    if opt.last_designer is None or not hasattr(opt.last_designer, "state_dict"):
        return _DesignerSnapshot(state=None, class_name=None, index=None)
    try:
        designer_state = opt.last_designer.state_dict(data=opt._data)
    except TypeError:
        designer_state = opt.last_designer.state_dict()
    designer_class = f"{type(opt.last_designer).__module__}.{type(opt.last_designer).__name__}"
    return _DesignerSnapshot(
        state=_pack_pickle(designer_state),
        class_name=designer_class,
        index=opt._last_designer_index,
    )


def _serialize_meta(
    opt: Any,
    *,
    designer_name: str,
    best_index: int | None,
    designer_class: str | None,
    designer_index: int | None,
) -> str:
    y_best = opt.y_best
    if isinstance(y_best, np.ndarray):
        y_best = y_best.tolist()

    meta: dict[str, Any] = {
        "version": 1,
        "designer_name": designer_name,
        "num_params": int(opt.num_params),
        "i_iter": int(opt._i_iter),
        "i_noise": int(opt._i_noise),
        "cum_dt_proposing": float(opt._cum_dt_proposing),
        "r_best_est": float(opt.r_best_est),
        "y_best": y_best,
        "best_index": best_index,
        "policy_init_present": _policy_init_flat(opt._policy_template) is not None,
        "best_params_present": opt.best_policy is not None,
    }
    if designer_class is not None:
        meta["designer_class"] = designer_class
    if designer_index is not None:
        meta["designer_index"] = int(designer_index)
    return json.dumps(meta)


def build_state_dict(opt: Any, *, designer_name: str) -> dict[str, np.ndarray]:
    obs = _serialize_observations(opt)
    trace = _serialize_trace(opt)
    best_params, best_index = _serialize_best(opt)
    designer = _serialize_designer(opt)
    meta_json = _serialize_meta(
        opt,
        designer_name=designer_name,
        best_index=best_index,
        designer_class=designer.class_name,
        designer_index=designer.index,
    )

    state: dict[str, np.ndarray] = {"meta": np.array(meta_json)}
    state.update(obs)
    state.update(trace)
    if designer.state is not None:
        state["designer_state"] = designer.state
    if best_params is not None:
        state["best_params"] = best_params
    policy_init = _policy_init_flat(opt._policy_template)
    if policy_init is not None:
        state["policy_init"] = policy_init
    return state


def save_optimizer_checkpoint(opt: Any, path: str, *, designer_name: str) -> None:
    state = build_state_dict(opt, designer_name=designer_name)
    np.savez_compressed(path, **state)


def load_optimizer_checkpoint(path: str) -> dict[str, Any]:
    with np.load(path) as data:
        state: dict[str, Any] = {k: data[k] for k in data.files}

    meta_raw = state["meta"].item() if hasattr(state["meta"], "item") else state["meta"]
    state["meta"] = json.loads(meta_raw)
    if "designer_state" in state:
        state["designer_state"] = _unpack_pickle(state["designer_state"])
    return state


def _restore_best_policy(opt: Any, state: dict[str, Any]) -> None:
    policy_init = state.get("policy_init")
    if policy_init is not None:
        _set_policy_baseline(opt._policy_template, policy_init)

    best_params = state.get("best_params")
    if best_params is None:
        opt.best_policy = opt._policy_template.clone()
        return

    best_policy = opt._policy_template.clone()
    _set_policy_baseline(best_policy, policy_init)
    best_policy.set_params(best_params)
    opt.best_policy = best_policy


def _restore_data(opt: Any, state: dict[str, Any]) -> None:
    params = state["params"]
    rreturn = state["rreturn"]
    rreturn_se = state["rreturn_se"]
    rreturn_est = state["rreturn_est"]

    policy_init = state.get("policy_init")
    opt._data = []
    for i in range(len(params)):
        policy = opt._policy_template.clone()
        _set_policy_baseline(policy, policy_init)
        policy.set_params(params[i])

        rr = rreturn[i]
        rr_se = None if np.all(np.isnan(rreturn_se[i])) else rreturn_se[i]
        rr_est = None if np.all(np.isnan(rreturn_est[i])) else rreturn_est[i]
        traj = Trajectory(
            rreturn=rr,
            states=np.empty((0,)),
            actions=np.empty((0,)),
            rreturn_se=rr_se,
            rreturn_est=rr_est,
        )
        opt._data.append(Datum(None, policy, None, traj))


def _restore_trace(opt: Any, state: dict[str, Any]) -> None:
    opt._trace = []
    for i in range(len(state["trace_rreturn"])):
        opt._trace.append(
            _TraceEntry(
                rreturn=float(state["trace_rreturn"][i]),
                rreturn_decision=float(state["trace_rreturn_decision"][i]),
                dt_prop=float(state["trace_dt_prop"][i]),
                dt_eval=float(state["trace_dt_eval"][i]),
            ),
        )


def _restore_meta_fields(opt: Any, meta: dict[str, Any]) -> None:
    opt._i_iter = int(meta.get("i_iter", len(opt._trace)))
    opt._i_noise = int(meta.get("i_noise", 0))
    opt._cum_dt_proposing = float(meta.get("cum_dt_proposing", 0.0))
    opt.r_best_est = float(meta.get("r_best_est", -1e99))
    y_best = meta.get("y_best")
    opt.y_best = np.asarray(y_best, dtype=np.float64) if isinstance(y_best, list) else y_best

    best_index = meta.get("best_index")
    if best_index is not None and 0 <= int(best_index) < len(opt._data):
        opt.best_datum = opt._data[int(best_index)]
    else:
        opt.best_datum = None


def _restore_designer_state(opt: Any, state: dict[str, Any], meta: dict[str, Any]) -> None:
    designer_state = state.get("designer_state")
    if designer_state is None:
        return
    designer_class = meta.get("designer_class")
    designer_index = meta.get("designer_index")

    target = None
    if designer_index is not None and 0 <= int(designer_index) < len(opt._opt_designers):
        cand = opt._opt_designers[int(designer_index)]
        cand_class = f"{type(cand).__module__}.{type(cand).__name__}"
        if designer_class is None or cand_class == designer_class:
            target = cand
    if target is None:
        for cand in opt._opt_designers:
            cand_class = f"{type(cand).__module__}.{type(cand).__name__}"
            if designer_class is None or cand_class == designer_class:
                target = cand
                break
    if target is None or not hasattr(target, "load_state_dict"):
        return
    target.load_state_dict(designer_state, data=opt._data)
    opt.last_designer = target
    try:
        opt._last_designer_index = int(opt._opt_designers.index(target))
    except ValueError:
        opt._last_designer_index = None


def apply_state_dict(opt: Any, state: dict[str, Any], *, designer_name: str) -> None:
    meta = state["meta"]
    if meta.get("designer_name") not in (None, designer_name):
        raise ValueError(f"Checkpoint designer_name mismatch: {meta.get('designer_name')} != {designer_name}")
    _restore_best_policy(opt, state)
    opt.initialize(designer_name)
    _restore_data(opt, state)
    _restore_trace(opt, state)
    _restore_meta_fields(opt, meta)
    _restore_designer_state(opt, state, meta)
