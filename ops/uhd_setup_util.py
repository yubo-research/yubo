import math

import torch
from torch.utils.data import DataLoader

from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.uhd_driver import UHDDriver
from optimizer.uhd_enn_imputer import ENNImputerConfig, ENNMinusImputer
from optimizer.uhd_enn_seed_selector import ENNMuPlusSeedSelector, ENNSeedSelectConfig


def _action_dim(space):
    if hasattr(space, "shape") and space.shape:
        return int(space.shape[0])
    if hasattr(space, "n"):
        return int(space.n)
    raise ValueError(f"Cannot get action dim from {type(space).__name__}: {space}")


def _maybe_attach_enn(loop: UHDDriver, *, module, env, enabled: bool, cfg: ENNImputerConfig) -> None:
    if not enabled:
        return
    if not hasattr(env, "torch_env"):
        return
    if not isinstance(loop.perturbator, SparseGaussianPerturbator):
        return

    k = getattr(loop.perturbator, "_k", None)
    if k is not None:
        imputer = ENNMinusImputer(
            module=module,
            cfg=cfg,
            noise_nz_fn=lambda s, sig: loop.perturbator.sample_global_nz(seed=s, sigma=sig),
        )
        loop.set_enn(minus_imputer=imputer, seed_selector=None)
        return

    if str(cfg.target) != "mu_plus" or int(cfg.num_candidates) <= 1:
        return

    prob = float(getattr(loop.perturbator, "_prob", 1.0))
    selector = ENNMuPlusSeedSelector(
        module=module,
        perturbator=loop.perturbator,
        cfg=ENNSeedSelectConfig(
            d=int(cfg.d),
            s=int(cfg.s),
            jl_seed=int(cfg.jl_seed),
            k=int(cfg.k),
            fit_interval=int(cfg.fit_interval),
            warmup_real_obs=int(cfg.warmup_real_obs),
            num_candidates=int(cfg.num_candidates),
            noise_prob=prob,
            embedder=str(cfg.embedder),
            gather_t=int(cfg.gather_t),
        ),
    )
    loop.set_enn(minus_imputer=None, seed_selector=selector)


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_accuracy_fn(module, device):
    from torchvision import datasets

    from problems.mnist_env import _MNIST_ROOT, _mnist_transform

    test_dataset = datasets.MNIST(
        root=_MNIST_ROOT,
        train=False,
        download=True,
        transform=_mnist_transform(),
    )
    images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    def _accuracy_fn():
        module.eval()
        with torch.no_grad():
            preds = module(images).argmax(dim=1)
        module.train()
        return float((preds == labels).float().mean())

    return _accuracy_fn


def _preload_mnist_train_to_device(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    from problems.mnist_env import _MNIST_ROOT, _mnist_transform

    train_dataset = datasets.MNIST(
        root=_MNIST_ROOT,
        train=True,
        download=True,
        transform=_mnist_transform(),
    )
    loader = DataLoader(train_dataset, batch_size=8192, shuffle=False, drop_last=False)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    images = torch.cat(xs, dim=0).to(device)
    labels = torch.cat(ys, dim=0).to(device)
    return images, labels


def _parse_enn_cfg(enn: dict[str, object] | None) -> tuple[bool, ENNImputerConfig]:
    # Handle ENNConfig dataclass by converting to dict first
    if hasattr(enn, "__dataclass_fields__"):
        from ops.uhd_config_types import ENNConfig

        if isinstance(enn, ENNConfig):
            enn = {
                "enn_minus_impute": enn.minus_impute,
                "enn_d": enn.d,
                "enn_s": enn.s,
                "enn_jl_seed": enn.jl_seed,
                "enn_k": enn.k,
                "enn_fit_interval": enn.fit_interval,
                "enn_warmup_real_obs": enn.warmup_real_obs,
                "enn_refresh_interval": enn.refresh_interval,
                "enn_se_threshold": enn.se_threshold,
                "enn_target": enn.target,
                "enn_num_candidates": enn.num_candidates,
                "enn_select_interval": enn.select_interval,
                "enn_embedder": enn.embedder,
                "enn_gather_t": enn.gather_t,
            }
    enn = {} if enn is None else dict(enn)
    enabled = bool(enn.get("enn_minus_impute", False))
    cfg = ENNImputerConfig(
        d=int(enn.get("enn_d", 100)),
        s=int(enn.get("enn_s", 4)),
        jl_seed=int(enn.get("enn_jl_seed", 123)),
        k=int(enn.get("enn_k", 25)),
        fit_interval=int(enn.get("enn_fit_interval", 50)),
        warmup_real_obs=int(enn.get("enn_warmup_real_obs", 200)),
        refresh_interval=int(enn.get("enn_refresh_interval", 50)),
        se_threshold=float(enn.get("enn_se_threshold", 0.25)),
        target=str(enn.get("enn_target", "mu_minus")),
        num_candidates=int(enn.get("enn_num_candidates", 1)),
        select_interval=int(enn.get("enn_select_interval", 1)),
        embedder=str(enn.get("enn_embedder", "direction")),
        gather_t=int(enn.get("enn_gather_t", 64)),
    )
    return enabled, cfg


def _evaluate_gym_with_denoise(
    env_conf,
    policy,
    *,
    eval_seed: int,
    noise_seed_0: int,
    frozen: bool,
    num_denoise: int | None,
) -> tuple[float, float]:
    import numpy as np

    from optimizer.trajectories import collect_trajectory

    if num_denoise is None or int(num_denoise) <= 1:
        ns = noise_seed_0 if frozen else int(eval_seed) + noise_seed_0
        return float(collect_trajectory(env_conf, policy, noise_seed=ns).rreturn), 0.0

    base_seed = noise_seed_0 if frozen else int(eval_seed) + noise_seed_0
    rets: list[float] = []
    for j in range(int(num_denoise)):
        ns = int(base_seed + j)
        rets.append(float(collect_trajectory(env_conf, policy, noise_seed=ns).rreturn))
    mu = float(np.mean(rets))
    se = float(np.std(rets) / math.sqrt(len(rets)))
    return mu, se
