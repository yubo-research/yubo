from __future__ import annotations

import numpy as np


def test_mezo_tracks_legacy_real_and_predicted_bests_separately():
    from ops.vec_uhd import _new_mezo_state, _track_mezo_best

    objective = _Objective(dim=2)
    state = _new_mezo_state(objective)

    real_x = np.asarray([1.0, 0.0])
    state.last_mu = 1.0
    state.last_imputed = False
    _track_mezo_best(state, real_x)

    pred_x = np.asarray([2.0, 0.0])
    state.last_mu = 2.0
    state.last_imputed = True
    _track_mezo_best(state, pred_x)

    assert state.y_best == 2.0
    assert state.y_best_real == 1.0
    assert state.y_best_pred == 2.0
    np.testing.assert_allclose(state.best_x, pred_x)
    np.testing.assert_allclose(state.best_x_real, real_x)


def test_bszo_tracks_real_best_after_predicted_best():
    from ops.vec_uhd import _new_bszo_state, _track_bszo_best

    objective = _Objective(dim=2)
    state = _new_bszo_state(objective)

    pred_x = np.asarray([2.0, 0.0])
    state.last_imputed = True
    _track_bszo_best(state, pred_x, 2.0)

    real_x = np.asarray([1.0, 0.0])
    state.last_imputed = False
    _track_bszo_best(state, real_x, 1.0)

    assert state.y_best == 2.0
    assert state.y_best_real == 1.0
    assert state.y_best_pred == 2.0
    np.testing.assert_allclose(state.best_x, pred_x)
    np.testing.assert_allclose(state.best_x_real, real_x)


def test_best_source_suffix_only_when_enabled():
    from ops.vec_uhd import (
        _format_source_best_suffix,
        _new_mezo_state,
        _track_mezo_best,
    )

    objective = _Objective(dim=1)
    state = _new_mezo_state(objective)
    state.last_mu = 1.0
    _track_mezo_best(state, np.asarray([1.0]))

    assert _format_source_best_suffix(state, False) == ""
    assert _format_source_best_suffix(state, True) == " y_best_real = 1.0000 y_best_pred = N/A"


def test_noise_uses_dense_uhd_backend_by_default():
    from ops.exp_uhd import _parse_cfg
    from ops.vec_uhd import _noise

    cfg = _parse_cfg({"env_tag": "f:sphere-2d", "num_rounds": 1, "perturb": "dense"})
    objective = _Objective(dim=3)

    noise = _noise(objective, cfg, 7, x=np.ones(3, dtype=np.float64))

    assert objective.dense_calls == [(7, None, None)]
    assert objective.eggroll_calls == []
    np.testing.assert_allclose(noise, np.asarray([7.0, 8.0, 9.0]))


def test_noise_can_replace_dense_with_eggroll_backend():
    from ops.exp_uhd import _parse_cfg
    from ops.vec_uhd import _noise

    cfg = _parse_cfg(
        {
            "env_tag": "f:sphere-2d",
            "num_rounds": 1,
            "perturb": "eggroll",
            "eggroll_noiser": "eggroll",
            "eggroll_rank": 2,
            "eggroll_group_size": 4,
            "eggroll_freeze_nonlora": True,
        }
    )
    objective = _Objective(dim=3)
    x = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    noise = _noise(objective, cfg, 11, x=x)

    assert objective.dense_calls == []
    assert len(objective.eggroll_calls) == 1
    call = objective.eggroll_calls[0]
    np.testing.assert_allclose(call["x"], x)
    assert call["seed"] == 11
    assert call["noiser_name"] == "eggroll"
    assert call["rank"] == 2
    assert call["group_size"] == 4
    assert call["freeze_nonlora"] is True
    np.testing.assert_allclose(noise, np.asarray([-11.0, -12.0, -13.0]))


def test_simple_minus_impute_runs_with_point_imputer():
    from ops.exp_uhd import _parse_cfg
    from ops.vec_uhd import _run_simple

    cfg = _parse_cfg(
        {
            "env_tag": "f:sphere-2d",
            "num_rounds": 2,
            "optimizer": "simple",
            "enn_minus_impute": True,
            "be_num_probes": 1,
            "enn_warmup_real_obs": 100,
        }
    )
    objective = _Objective(dim=2)

    _run_simple(objective, cfg)

    assert len(objective.evals) >= 2
    assert objective.policies


class _Objective:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.x0 = np.zeros(self.dim, dtype=np.float64)
        self.dense_calls = []
        self.eggroll_calls = []
        self.evals = []
        self.policies = []

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        self.dense_calls.append((int(seed), num_dim_target, num_module_target))
        return np.arange(int(seed), int(seed) + self.dim, dtype=np.float64)

    def sample_eggroll_noiser_noise(
        self,
        x: np.ndarray,
        *,
        seed: int,
        noiser_name: str,
        rank: int,
        group_size: int,
        freeze_nonlora: bool,
    ) -> np.ndarray:
        self.eggroll_calls.append(
            {
                "x": np.asarray(x, dtype=np.float64).copy(),
                "seed": int(seed),
                "noiser_name": str(noiser_name),
                "rank": int(rank),
                "group_size": int(group_size),
                "freeze_nonlora": bool(freeze_nonlora),
            }
        )
        return -np.arange(int(seed), int(seed) + self.dim, dtype=np.float64)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def embed_many(self, xs: np.ndarray) -> np.ndarray:
        return np.asarray(xs, dtype=np.float64)

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        self.evals.append((np.asarray(x, dtype=np.float64).copy(), int(seed)))
        return -float(np.linalg.norm(x)), 0.0

    def make_policy(self, x: np.ndarray) -> np.ndarray:
        self.policies.append(np.asarray(x, dtype=np.float64).copy())
        return np.asarray(x, dtype=np.float64)
