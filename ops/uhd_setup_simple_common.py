import time

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


def _try_make_np_policy(env_conf):
    if env_conf.policy_class is None:
        return None
    cand = env_conf.policy_class(env_conf)
    if isinstance(cand, torch.nn.Module):
        return None
    if not hasattr(cand, "get_params"):
        return None
    return cand


def _make_gym_policy(env_conf, device, num_state, num_action):
    import warnings

    from problems.mlp_torch_policy import MLPPolicyModule
    from problems.torch_policy import TorchPolicy

    if env_conf.policy_class is not None:
        cand = env_conf.policy_class(env_conf)
        if isinstance(cand, torch.nn.Module):
            return cand.to(device), cand
        warnings.warn(
            f"Non-module policy_class {env_conf.policy_class}; using MLPPolicyModule.",
            stacklevel=2,
        )
    module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
    return module, TorchPolicy(module, env_conf)


def _make_perturbator(module, num_dim_target):
    if num_dim_target is not None:
        return SparseGaussianPerturbator(module, num_dim_target=num_dim_target)
    return GaussianPerturbator(module)


def _run_simple_iterations(
    uhd,
    *,
    evaluate_fn,
    accuracy_fn,
    num_rounds: int,
    log_interval: int,
    accuracy_interval: int,
    target_accuracy: float | None,
) -> None:
    t0 = time.perf_counter()
    acc = None
    for i in range(num_rounds):
        uhd.ask()
        mu, se = evaluate_fn()
        uhd.tell(mu, se)

        if not _should_log_simple(i, num_rounds, log_interval):
            continue
        y_best = uhd.y_best
        y_str = f"{y_best:.4f}" if y_best is not None else "N/A"
        if accuracy_fn is not None and (acc is None or i == num_rounds - 1 or (accuracy_interval > 0 and i % accuracy_interval == 0)):
            acc = accuracy_fn()
        line = f"EVAL: i_iter = {i} sigma = {uhd.sigma:.6f} mu = {mu:.4f} se = {se:.4f} y_best = {y_str}"
        if acc is not None:
            line += f" test_acc = {acc:.4f}"
        print(line)
        if target_accuracy is not None and mu >= target_accuracy:
            elapsed = time.perf_counter() - t0
            print(f"UHD-Simple: target reached {mu:.4f} >= {target_accuracy:.4f} at i_iter={i} ({elapsed:.2f}s)")
            break

    elapsed = time.perf_counter() - t0
    print(f"UHD-Simple: elapsed = {elapsed:.2f}s ({min(i + 1, num_rounds)} iterations)")


def _eval_full_train_acc(module, images, labels, chunk: int) -> tuple[float, float]:
    was_training = module.training
    module.eval()
    correct = _count_correct(module, images, labels, chunk)
    if was_training:
        module.train()
    n = int(images.shape[0])
    acc = correct / n
    se = float((acc * (1.0 - acc) / n) ** 0.5)
    return float(acc), se


def _count_correct(module, images, labels, chunk: int) -> int:
    n = int(images.shape[0])
    correct = 0
    with torch.inference_mode():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            pred = module(images[start:end]).argmax(dim=1)
            correct += int((pred == labels[start:end]).sum())
    return correct


def _should_log_simple(i: int, num_rounds: int, log_interval: int) -> bool:
    return i == 0 or i == num_rounds - 1 or (log_interval > 0 and i % log_interval == 0)
