import torch

from ops.uhd_config import BEConfig
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator


def _make_simple_optimizer(
    module,
    perturbator,
    *,
    optimizer: str,
    sigma: float,
    dim: int,
    embed_module=None,
    embed_bounds=None,
    be: BEConfig | None = None,
):
    cfg = be if be is not None else BEConfig()
    if optimizer in {"simple_be", "mezo_be"}:
        from embedding.behavioral_embedder import BehavioralEmbedder

        if embed_bounds is None:
            embed_bounds = _mnist_embed_bounds()
        if embed_module is None:
            embed_module = module
        embedder = BehavioralEmbedder(embed_bounds, num_probes=cfg.num_probes, seed=0)

        if optimizer == "mezo_be":
            from optimizer.uhd_simple_be import UHDMeZOBE

            return UHDMeZOBE(
                perturbator,
                dim,
                embed_module,
                embedder,
                sigma=sigma,
                num_candidates=cfg.num_candidates,
                warmup=cfg.warmup,
                fit_interval=cfg.fit_interval,
                enn_k=cfg.enn_k,
            )

        from optimizer.uhd_simple_be import UHDSimpleBE

        return UHDSimpleBE(
            perturbator,
            sigma_0=sigma,
            dim=dim,
            module=embed_module,
            embedder=embedder,
            num_candidates=cfg.num_candidates,
            warmup=cfg.warmup,
            fit_interval=cfg.fit_interval,
            enn_k=cfg.enn_k,
            sigma_range=cfg.sigma_range,
        )
    from optimizer.uhd_simple import UHDSimple

    return UHDSimple(perturbator, sigma_0=sigma, dim=dim, sigma_range=cfg.sigma_range)


def _mnist_embed_bounds() -> torch.Tensor:
    lb = (0.0 - 0.1307) / 0.3081
    ub = (1.0 - 0.1307) / 0.3081
    bounds = torch.zeros(2, 1, 28, 28)
    bounds[0] = lb
    bounds[1] = ub
    return bounds


def _gym_embed_bounds(num_state: int) -> torch.Tensor:
    bounds = torch.zeros(2, num_state)
    bounds[0] = -1.0
    bounds[1] = 1.0
    return bounds


def _make_perturbator(module, num_dim_target):
    if num_dim_target is not None:
        return SparseGaussianPerturbator(module, num_dim_target=num_dim_target)
    return GaussianPerturbator(module)


__all__ = [
    "_gym_embed_bounds",
    "_make_perturbator",
    "_make_simple_optimizer",
    "_mnist_embed_bounds",
]
