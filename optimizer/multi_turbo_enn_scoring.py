from __future__ import annotations

import numpy as np


def _as_2d(mu, sigma) -> tuple[np.ndarray, np.ndarray]:
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim == 1:
        mu_arr = mu_arr.reshape(-1, 1)
    if sigma is None:
        sigma_arr = np.zeros_like(mu_arr, dtype=float)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 1:
            sigma_arr = sigma_arr.reshape(-1, 1)
    if mu_arr.shape != sigma_arr.shape:
        raise RuntimeError(f"mu/sigma shape mismatch: {mu_arr.shape} vs {sigma_arr.shape}")
    return mu_arr, sigma_arr


def _scalarize_values(
    y,
    *,
    tr_state,
    allow_random: bool,
    region_rng: np.random.Generator,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        return y_arr.reshape(-1)
    if y_arr.ndim != 2:
        raise RuntimeError(f"unexpected candidate value shape: {y_arr.shape}")
    if y_arr.shape[1] == 1:
        return y_arr[:, 0]
    if tr_state is None or not hasattr(tr_state, "scalarize"):
        if allow_random:
            return region_rng.random(size=(y_arr.shape[0],))
        raise RuntimeError("multi-objective scoring requires scalarize support")
    scalar = tr_state.scalarize(y_arr, clip=False)
    scalar = np.asarray(scalar, dtype=float).reshape(-1)
    if scalar.shape[0] != y_arr.shape[0]:
        raise RuntimeError(f"scalarized scores shape {scalar.shape} != {y_arr.shape[0]}")
    return scalar


def _acq_mode_from_name(acq_name: str) -> str:
    if "thompson" in acq_name:
        return "thompson"
    if "ucb" in acq_name:
        return "ucb"
    if "pareto" in acq_name:
        return "pareto"
    return "unknown"


def _score_with_turbo(
    child_designer,
    *,
    x_region: np.ndarray,
    allow_random: bool,
    region_rng: np.random.Generator,
) -> np.ndarray | None:
    turbo = getattr(child_designer, "_turbo", None)
    if turbo is None:
        return None
    surrogate = getattr(turbo, "_surrogate", None)
    tr_state = getattr(turbo, "_tr_state", None)
    acq_optimizer = getattr(turbo, "_acq_optimizer", None)
    if surrogate is None or acq_optimizer is None:
        return None

    acq_mode = _acq_mode_from_name(type(acq_optimizer).__name__.lower())
    if acq_mode == "thompson":
        samples = surrogate.sample(x_region, 1, region_rng)
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 3 and samples.shape[0] >= 1:
            return _scalarize_values(
                samples[0],
                tr_state=tr_state,
                allow_random=allow_random,
                region_rng=region_rng,
            )
        if samples.ndim == 2:
            return _scalarize_values(
                samples,
                tr_state=tr_state,
                allow_random=allow_random,
                region_rng=region_rng,
            )
        return None

    posterior = surrogate.predict(x_region)
    mu_arr, sigma_arr = _as_2d(posterior.mu, getattr(posterior, "sigma", None))
    if acq_mode == "ucb":
        beta = float(getattr(acq_optimizer, "_beta", 1.0))
        return _scalarize_values(
            mu_arr + beta * sigma_arr,
            tr_state=tr_state,
            allow_random=allow_random,
            region_rng=region_rng,
        )
    if acq_mode == "pareto":
        if mu_arr.shape[1] == 1:
            values = np.column_stack([mu_arr[:, 0], sigma_arr[:, 0]])
        else:
            values = mu_arr
        from nds import ndomsort

        fronts = ndomsort.non_domin_sort(-values, only_front_indices=True)
        return -np.asarray(fronts, dtype=float)
    return None


def _score_with_predict(
    child_designer,
    *,
    x_region: np.ndarray,
    acq_type: str,
    allow_random: bool,
    region_rng: np.random.Generator,
) -> np.ndarray:
    predict = getattr(child_designer, "predict_mu_sigma", None)
    if callable(predict):
        mu_sigma = predict(x_region)
    else:
        mu_sigma = None
    if mu_sigma is None:
        best_val = 0.0
        best_datum = child_designer.best_datum()
        if best_datum is not None:
            best_val = float(best_datum.trajectory.get_decision_rreturn())
        return best_val + region_rng.random(size=(x_region.shape[0],))

    mu_arr, sigma_arr = _as_2d(mu_sigma[0], mu_sigma[1])
    if mu_arr.shape[0] != x_region.shape[0]:
        raise RuntimeError(f"mu rows {mu_arr.shape[0]} != candidates {x_region.shape[0]}")
    if mu_arr.shape[1] == 1:
        if acq_type in ("thompson", "draw"):
            return mu_arr[:, 0] + sigma_arr[:, 0] * region_rng.normal(size=mu_arr.shape[0])
        return mu_arr[:, 0] + sigma_arr[:, 0]

    if acq_type in ("thompson", "draw"):
        y = mu_arr + sigma_arr * region_rng.normal(size=mu_arr.shape)
    else:
        y = mu_arr + sigma_arr
    scalarize = getattr(child_designer, "scalarize", None)
    tr_state = child_designer._turbo._tr_state if getattr(child_designer, "_turbo", None) is not None else None
    if callable(scalarize):
        scalar = np.asarray(scalarize(y, clip=False), dtype=float).reshape(-1)
        if scalar.shape[0] != mu_arr.shape[0]:
            raise RuntimeError(f"scalarized scores shape {scalar.shape} != {mu_arr.shape[0]}")
        return scalar
    return _scalarize_values(y, tr_state=tr_state, allow_random=allow_random, region_rng=region_rng)


def _score_region_candidates(
    child_designer,
    *,
    x_region: np.ndarray,
    acq_type: str,
    allow_random: bool,
    region_rng: np.random.Generator,
) -> np.ndarray:
    scores = _score_with_turbo(
        child_designer,
        x_region=x_region,
        allow_random=allow_random,
        region_rng=region_rng,
    )
    if scores is None:
        scores = _score_with_predict(
            child_designer,
            x_region=x_region,
            acq_type=acq_type,
            allow_random=allow_random,
            region_rng=region_rng,
        )
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape[0] != x_region.shape[0]:
        raise RuntimeError(f"score rows {scores.shape[0]} != candidates {x_region.shape[0]}")
    return scores


def score_multi_candidates(
    x_all: np.ndarray,
    region_indices: list[int],
    *,
    child_designers: list,
    region_data_lens: list[int],
    region_rngs: list[np.random.Generator],
    acq_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    scores = np.full((x_all.shape[0],), np.nan, dtype=float)
    per_region: list[list[int]] = [[] for _ in range(len(child_designers))]
    for idx, region_idx in enumerate(region_indices):
        per_region[region_idx].append(idx)
    for region_idx, indices in enumerate(per_region):
        if not indices:
            continue
        idx_arr = np.asarray(indices, dtype=int)
        allow_random = region_data_lens[region_idx] == 0
        scores[idx_arr] = _score_region_candidates(
            child_designers[region_idx],
            x_region=x_all[idx_arr],
            acq_type=acq_type,
            allow_random=allow_random,
            region_rng=region_rngs[region_idx],
        )
    nan_mask = ~np.isfinite(scores)
    if np.any(nan_mask):
        scores[nan_mask] = rng.random(size=int(np.sum(nan_mask)))
    return scores + rng.uniform(0.0, 1e-12, size=scores.shape)
