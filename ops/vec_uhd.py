from __future__ import annotations

from ops.uhd_config import UHDConfig
from ops.vec_uhd_bszo import _run_bszo
from ops.vec_uhd_mezo import _run_mezo
from ops.vec_uhd_simple import _run_simple


_BE_OPTIMIZERS = {"simple_be", "mezo_be", "bszo_be"}


def run_uhd_vector_loop(cfg: UHDConfig) -> None:
    optimizer = str(cfg.optimizer)
    _validate_minus_impute_optimizer(cfg)
    if cfg.enn.minus_impute and int(cfg.be.num_probes) < 1:
        raise ValueError("UHD vector enn_minus_impute requires be_num_probes >= 1 for behavior embeddings.")

    from problems.uhd_obj import build_uhd_vector_objective

    embed_num_probes = cfg.be.num_probes if optimizer in _BE_OPTIMIZERS or cfg.enn.minus_impute else 0
    built = build_uhd_vector_objective(cfg, embed_num_probes=embed_num_probes)
    objective = built.objective

    print(
        "UHD-Vector: "
        f"source = {built.source} env_tag = {cfg.env_tag} optimizer = {optimizer} dim = {objective.dim} "
        f"perturb = {cfg.perturb_backend} "
        f"steps_per_episode = {objective.steps_per_episode} num_envs = {objective.num_envs}"
    )
    try:
        if optimizer in {"simple", "simple_be"}:
            _run_simple(objective, cfg)
        elif optimizer in {"mezo", "mezo_be"}:
            _run_mezo(objective, cfg)
        elif optimizer in {"bszo", "bszo_be"}:
            _run_bszo(objective, cfg)
        else:
            raise ValueError(f"Unknown UHD vector optimizer: {optimizer}")
    finally:
        close = getattr(objective, "close", None)
        if callable(close):
            close()


def _validate_minus_impute_optimizer(cfg: UHDConfig) -> None:
    if not cfg.enn.minus_impute or str(cfg.optimizer) in {
        "simple",
        "simple_be",
        "mezo",
        "mezo_be",
        "bszo",
        "bszo_be",
    }:
        return
    raise ValueError("UHD vector enn_minus_impute is currently supported for optimizer='simple', 'simple_be', 'mezo', 'mezo_be', 'bszo', and 'bszo_be'.")
