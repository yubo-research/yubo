from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JittedFnConfig:
    env_adapter: Any
    noiser: Any
    frozen_noiser_params: Any
    frozen_params: Any
    es_tree_key: Any
    es_map: Any
    rank_transform: bool
    deterministic_policy: bool


def rank_scores(jnp, raw_scores, rank_transform: bool):
    if not rank_transform:
        return raw_scores
    population = raw_scores.shape[0]
    ranks = jnp.argsort(jnp.argsort(raw_scores)).astype(jnp.float32)
    return ranks / jnp.maximum(float(population - 1), 1.0)


def apply_noiser_update(jnp, cfg: JittedFnConfig, noiser_params, params, raw_scores, epoch):
    population = raw_scores.shape[0]
    iterinfo = (
        jnp.full((population,), epoch, dtype=jnp.int32),
        jnp.arange(population, dtype=jnp.int32),
    )
    fitnesses = cfg.noiser.convert_fitnesses(
        cfg.frozen_noiser_params,
        noiser_params,
        rank_scores(jnp, raw_scores, cfg.rank_transform),
    )
    return cfg.noiser.do_updates(
        cfg.frozen_noiser_params,
        noiser_params,
        params,
        cfg.es_tree_key,
        fitnesses,
        iterinfo,
        cfg.es_map,
    )
