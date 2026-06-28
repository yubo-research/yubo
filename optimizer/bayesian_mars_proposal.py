from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mars_basis import _HingeFactor, _MarsTerm, _screen_features
from .mars_config import BayesianMarsSurrogateConfig
from .mars_fit import _candidate_knots_by_feature


@dataclass(frozen=True)
class _TermProposal:
    knots: dict[int, np.ndarray]
    features: tuple[int, ...]
    interaction_order: int

    @classmethod
    def from_inputs(cls, x: np.ndarray, y_std: np.ndarray, cfg: BayesianMarsSurrogateConfig) -> _TermProposal:
        features = _screen_features(x, y_std, cfg.basis.feature_screen)
        knots = {feature: values for feature, values in _candidate_knots_by_feature(x, features, cfg.basis).items() if values.size}
        return cls(knots=knots, features=tuple(sorted(knots)), interaction_order=int(cfg.basis.interaction_order))

    def has_terms(self) -> bool:
        return bool(self.features)

    def sample_excluding(self, excluded: set[_MarsTerm], rng: np.random.Generator) -> _MarsTerm | None:
        for _ in range(10000):
            term = self.sample(rng)
            if term is not None and term not in excluded:
                return term
        return None

    def sample(self, rng: np.random.Generator) -> _MarsTerm | None:
        if not self.features:
            return None
        if self._sample_degree(rng) == 1:
            return _MarsTerm((self._sample_factor(int(rng.choice(self.features)), rng),))
        return self._sample_interaction(rng)

    def log_prob_excluding(self, term: _MarsTerm, excluded: set[_MarsTerm]) -> float:
        if term in excluded:
            return -math.inf
        base = self.log_prob(term)
        mass = self.excluded_mass(excluded)
        if not np.isfinite(base) or mass >= 1.0:
            return -math.inf
        return float(base - math.log1p(-mass))

    def excluded_mass(self, excluded: set[_MarsTerm]) -> float:
        return float(sum(math.exp(value) for value in (self.log_prob(term) for term in excluded) if np.isfinite(value)))

    def log_prob(self, term: _MarsTerm) -> float:
        factors = tuple(term.factors)
        if len(factors) == 1:
            return self._log_prob_main(factors[0])
        if len(factors) == 2 and self._allows_interactions():
            return self._log_prob_interaction(factors)
        return -math.inf

    def _sample_degree(self, rng: np.random.Generator) -> int:
        if not self._allows_interactions():
            return 1
        return 1 if float(rng.random()) < 0.5 else 2

    def _sample_interaction(self, rng: np.random.Generator) -> _MarsTerm | None:
        if not self._allows_interactions():
            return None
        features = tuple(sorted(int(feature) for feature in rng.choice(self.features, size=2, replace=False)))
        factors = tuple(self._sample_factor(feature, rng) for feature in features)
        return _canonical_term(factors)

    def _sample_factor(self, feature: int, rng: np.random.Generator) -> _HingeFactor:
        knots = self.knots[int(feature)]
        knot = float(knots[int(rng.integers(0, knots.size))])
        side = 1 if float(rng.random()) < 0.5 else -1
        return _HingeFactor(int(feature), knot, side)

    def _log_prob_main(self, factor: _HingeFactor) -> float:
        if not self._valid_factor(factor):
            return -math.inf
        return math.log(self._degree_prob(1)) - math.log(len(self.features)) - math.log(self.knots[int(factor.feature)].size) - math.log(2.0)

    def _log_prob_interaction(self, factors: tuple[_HingeFactor, ...]) -> float:
        if len({int(factor.feature) for factor in factors}) != 2:
            return -math.inf
        if not all(self._valid_factor(factor) for factor in factors):
            return -math.inf
        base = math.log(self._degree_prob(2)) - math.log(math.comb(len(self.features), 2))
        return float(base + sum(-math.log(self.knots[int(factor.feature)].size) - math.log(2.0) for factor in factors))

    def _valid_factor(self, factor: _HingeFactor) -> bool:
        knots = self.knots.get(int(factor.feature))
        if knots is None or int(factor.side) not in (-1, 1):
            return False
        return bool(np.any(np.isclose(knots, float(factor.knot), rtol=0.0, atol=1e-12)))

    def _degree_prob(self, degree: int) -> float:
        if degree == 1:
            return 0.5 if self._allows_interactions() else 1.0
        return 0.5 if self._allows_interactions() else 0.0

    def _allows_interactions(self) -> bool:
        return self.interaction_order >= 2 and len(self.features) >= 2


def _canonical_term(factors: tuple[_HingeFactor, ...]) -> _MarsTerm:
    return _MarsTerm(tuple(sorted(factors, key=lambda factor: (int(factor.feature), float(factor.knot), int(factor.side)))))
