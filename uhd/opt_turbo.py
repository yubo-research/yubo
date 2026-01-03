from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn

from uhd.param_accessor import make_param_accessor
from uhd.uhd_collector import UHDCollector


@dataclass
class UHDBOConfig:
    num_candidates: int
    perturber: Any
    embedder: Any
    selector: Any
    metric: Any
    num_trail: int = 32


@dataclass
class TurboObservation:
    y: float
    embedding: torch.Tensor


def _trim_history(history: List[TurboObservation], best: TurboObservation, limit: int) -> None:
    assert isinstance(limit, int) and limit > 0
    assert any(obs is best for obs in history)
    excess = len(history) - limit
    idx = 0
    while excess > 0:
        if history[idx] is best:
            idx += 1
            if idx >= len(history):
                idx = 0
            continue
        history.pop(idx)
        excess -= 1
        if idx >= len(history):
            idx = 0


def optimize_turbo(controller: nn.Module, collector: UHDCollector, num_rounds: int, config: UHDBOConfig) -> float:
    assert isinstance(controller, nn.Module)
    assert isinstance(collector, UHDCollector)
    assert isinstance(num_rounds, int) and num_rounds >= 0
    assert isinstance(config, UHDBOConfig)
    assert isinstance(config.num_candidates, int) and config.num_candidates > 0
    assert isinstance(config.num_trail, int) and config.num_trail > 0
    assert hasattr(config.perturber, "perturb")
    assert hasattr(config.perturber, "unperturb")
    assert hasattr(config.perturber, "incorporate")
    assert hasattr(config.embedder, "embed")
    assert hasattr(config.selector, "select")
    assert hasattr(config.metric, "measure")

    accessor = make_param_accessor(controller)
    collector(f"num_dim = {accessor.numel()}")

    if num_rounds == 0:
        collector("DONE")
        initial_y = float(config.metric.measure(controller))
        return initial_y

    initial_y = float(config.metric.measure(controller))
    initial_embedding = config.embedder.embed(accessor.clone_flat())
    assert isinstance(initial_embedding, torch.Tensor)
    collector.update_best(initial_y)
    history: List[TurboObservation] = [TurboObservation(y=initial_y, embedding=initial_embedding)]
    best_observation = history[0]
    iteration_index = 0

    for _ in range(num_rounds):
        collector.start_prop()
        ys = [obs.y for obs in history]
        candidate_embeddings: List[torch.Tensor] = []
        if config.num_candidates == 1:
            rng_state = torch.random.get_rng_state()
            config.perturber.perturb(accessor, ys)
            emb = config.embedder.embed(accessor.clone_flat())
            assert isinstance(emb, torch.Tensor)
            candidate_embeddings.append(emb)
            config.perturber.unperturb(accessor)
            idx = 0
            collector.stop_prop()
            torch.random.set_rng_state(rng_state)
            config.perturber.perturb(accessor, ys)
        else:
            base_seed = int(torch.random.initial_seed() + iteration_index * 1000003)
            for candidate_idx in range(config.num_candidates):
                torch.manual_seed(base_seed + candidate_idx)
                config.perturber.perturb(accessor, ys)
                emb = config.embedder.embed(accessor.clone_flat())
                assert isinstance(emb, torch.Tensor)
                candidate_embeddings.append(emb)
                config.perturber.unperturb(accessor)

            idx = config.selector.select(candidate_embeddings)
            assert isinstance(idx, int) and 0 <= idx < len(candidate_embeddings)
            collector.stop_prop()
            torch.manual_seed(base_seed + idx)
            config.perturber.perturb(accessor, ys)

        collector.start_eval()
        y_new: float = float(config.metric.measure(controller))
        collector.stop_eval()
        if hasattr(config.perturber, "tr_update"):
            config.perturber.tr_update(y_new)
        observation = TurboObservation(y=y_new, embedding=candidate_embeddings[idx])
        history.append(observation)
        if y_new > best_observation.y:
            best_observation = observation
            config.perturber.incorporate(accessor)
        else:
            config.perturber.unperturb(accessor)
        if len(history) > config.num_trail:
            _trim_history(history, best_observation, config.num_trail)

        collector.params(controller)
        collector.trace(y_new)
        iteration_index += 1

    collector("DONE")
    return best_observation.y
