from __future__ import annotations

import numpy as np


class _VectorObjective:
    _vectorize = True

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        self.calls: list[tuple[str, int, int]] = []

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(x_batch.shape[0])
        self.calls.append(("many", int(seed), n))
        means = self._scores[:n]
        ses = np.zeros_like(means)
        return means, ses


def test_be_pick_candidate_uses_evaluate_many():
    from ops.vec_uhd_be import be_pick_candidate

    objective = _VectorObjective(np.asarray([0.1, 0.9, 0.3]))
    candidates = np.zeros((3, 2))
    pick = be_pick_candidate(objective, candidates, seed=7)
    assert pick is not None
    best, mu, se = pick
    assert best == 1
    assert mu == 0.9
    assert se == 0.0
    assert objective.calls == [("many", 7, 3)]


def test_be_pick_candidate_skips_without_vectorize():
    from ops.vec_uhd_be import be_pick_candidate

    objective = _VectorObjective(np.asarray([1.0]))
    objective._vectorize = False
    assert be_pick_candidate(objective, np.zeros((1, 2)), seed=0) is None
    assert objective.calls == []


def test_be_pick_mezo_seed_picks_largest_gradient_signal():
    from ops.vec_uhd_be import be_pick_mezo_seed

    objective = _VectorObjective(np.asarray([0.1, 2.0, 0.5]))
    objective._minus_scores = np.asarray([0.0, 0.0, 0.5])

    def evaluate_many(x_batch, *, seed):
        n = int(x_batch.shape[0])
        objective.calls.append(("many", int(seed), n))
        if int(seed) == 0:
            return objective._scores[:n], np.zeros(n)
        return objective._minus_scores[:n], np.zeros(n)

    objective.evaluate_many = evaluate_many  # type: ignore[method-assign]

    x_plus = np.zeros((3, 2))
    x_minus = np.zeros((3, 2))
    best = be_pick_mezo_seed(objective, x_plus, x_minus, [0, 1, 2], sigma=0.1)
    assert best == 1
    assert len(objective.calls) == 2


def test_be_pick_mezo_seed_skips_without_vectorize():
    from ops.vec_uhd_be import be_pick_mezo_seed

    objective = _VectorObjective(np.asarray([1.0]))
    objective._vectorize = False
    assert be_pick_mezo_seed(objective, np.zeros((1, 2)), np.zeros((1, 2)), [0], sigma=0.1) is None
