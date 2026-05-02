import math
import warnings

import torch
import torch.nn.functional as F

from ops.uhd_config import BEConfig
from ops.uhd_setup_monolith_opt import (
    _gym_embed_bounds,
    _make_perturbator,
    _make_simple_optimizer,
)
from ops.uhd_setup_monolith_support import (
    _action_dim,
    _evaluate_gym_with_denoise,
    _get_device,
    _make_accuracy_fn,
    _make_torch_env,
    _preload_mnist_train_to_device,
)


def _run_simple_torch(
    env,
    env_runtime,
    env_tag,
    num_rounds,
    *,
    optimizer,
    sigma,
    num_dim_target,
    batch_size,
    log_interval,
    accuracy_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    device = _get_device()
    torch_env = env.torch_env()
    module = torch_env.module.to(device)
    module.train()
    train_images, train_labels = _preload_mnist_train_to_device(device)

    dim = sum(p.numel() for p in module.parameters())
    perturbator = _make_perturbator(module, num_dim_target)
    uhd = _make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        be=be,
    )
    accuracy_fn = _make_accuracy_fn(module, device)

    if str(env_tag) == "mnist_fulltrain_acc":

        def evaluate_fn():
            return _eval_full_train_acc(module, train_images, train_labels, 8192)
    else:

        def evaluate_fn():
            if num_denoise is None or int(num_denoise) <= 1:
                g = torch.Generator()
                g.manual_seed(int(uhd.eval_seed))
                idx = torch.randint(train_images.shape[0], (batch_size,), generator=g).to(device)
                with torch.inference_mode():
                    logits = module(train_images.index_select(0, idx))
                    per_sample = F.cross_entropy(logits, train_labels.index_select(0, idx), reduction="none")
                mu = -float(per_sample.mean())
                se = float(per_sample.std() / math.sqrt(len(per_sample)))
                return mu, se

            vals = []
            for j in range(int(num_denoise)):
                g = torch.Generator()
                g.manual_seed(int(uhd.eval_seed + j))
                idx = torch.randint(train_images.shape[0], (batch_size,), generator=g).to(device)
                with torch.inference_mode():
                    logits = module(train_images.index_select(0, idx))
                    per_sample = F.cross_entropy(logits, train_labels.index_select(0, idx), reduction="none")
                vals.append(-float(per_sample.mean()))
            mu = float(sum(vals) / len(vals))
            se = float((torch.tensor(vals, dtype=torch.float64).std(unbiased=False) / math.sqrt(len(vals))).item())
            return mu, se

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}")
    _run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=accuracy_fn,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
    )


def _run_simple_gym(
    env,
    problem,
    env_tag,
    num_rounds,
    *,
    optimizer,
    sigma,
    num_dim_target,
    log_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    from common.seed_all import seed_all

    env_runtime = problem.env
    env.close()
    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed) + 27)

    np_policy = _try_make_np_policy(problem)
    if np_policy is not None:
        _run_simple_gym_np(
            np_policy,
            env_runtime,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            log_interval=log_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )
        return

    torch_env_wrapper = _make_torch_env(problem)
    if hasattr(torch_env_wrapper, "torch_env"):
        _run_simple_gym_torch(
            problem,
            env_tag,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            num_dim_target=num_dim_target,
            log_interval=log_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )
        return

    device = torch.device("cpu")
    env_runtime.ensure_spaces()
    num_state = env_runtime.gym_conf.state_space.shape[0]
    num_action = _action_dim(env_runtime.action_space)
    noise_seed_0 = env_runtime.noise_seed_0 or 0

    module, policy = _make_gym_policy(problem, device, num_state, num_action)
    dim = sum(p.numel() for p in module.parameters())
    perturbator = _make_perturbator(module, num_dim_target)

    embed_module = getattr(module, "model", module)
    embed_bounds = _gym_embed_bounds(num_state)
    uhd = _make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        embed_module=embed_module,
        embed_bounds=embed_bounds,
        be=be,
    )

    frozen = bool(getattr(env_runtime, "frozen_noise", False))

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_runtime,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}, state={num_state}, action={num_action}")
    _run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
    )


def _try_make_np_policy(problem):
    """Try to create a numpy-based policy from the problem.

    Returns the policy if it's numpy-based (has get_params method), else None.
    """
    cand = problem.build_policy()
    if isinstance(cand, torch.nn.Module):
        return None
    if not hasattr(cand, "get_params"):
        return None
    return cand


def _run_simple_gym_np(
    policy,
    env_runtime,
    num_rounds,
    *,
    optimizer,
    sigma,
    log_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    from embedding.behavioral_embedder import BehavioralEmbedder
    from optimizer.uhd_mezo_np import UHDMeZOBENp, UHDMeZONp
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    noise_seed_0 = env_runtime.noise_seed_0 or 0
    frozen = bool(getattr(env_runtime, "frozen_noise", False))
    dim = policy.num_params()
    param_clip = (-1.0, 1.0)

    cfg = be if be is not None else BEConfig()
    if optimizer in {"simple_be", "mezo_be"}:
        num_state = env_runtime.gym_conf.state_space.shape[0]
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        if optimizer == "simple_be":
            uhd = UHDSimpleBENp(
                policy,
                embedder,
                sigma_0=sigma,
                param_clip=param_clip,
                num_candidates=cfg.num_candidates,
                warmup=cfg.warmup,
                fit_interval=cfg.fit_interval,
                enn_k=cfg.enn_k,
            )
        else:
            uhd = UHDMeZOBENp(
                policy,
                embedder,
                sigma=sigma,
                lr=0.001,
                param_clip=param_clip,
                num_candidates=cfg.num_candidates,
                warmup=cfg.warmup,
                fit_interval=cfg.fit_interval,
                enn_k=cfg.enn_k,
            )
    elif optimizer == "mezo":
        uhd = UHDMeZONp(policy, sigma=sigma, lr=0.001, param_clip=param_clip)
    else:
        uhd = UHDSimpleNp(policy, sigma_0=sigma, param_clip=param_clip)

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_runtime,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Np: num_params = {dim}, optimizer = {optimizer}")
    _run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
    )


def _run_simple_gym_torch(
    problem,
    env_tag,
    num_rounds,
    *,
    optimizer,
    sigma,
    num_dim_target,
    log_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    """Run simple loop for gym environments with torch MLP policies using in-place perturbation."""
    from common.seed_all import seed_all

    env_runtime = problem.env
    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed) + 27)

    device = torch.device("cpu")
    env_runtime.ensure_spaces()
    num_state = env_runtime.gym_conf.state_space.shape[0]
    num_action = _action_dim(env_runtime.action_space)
    noise_seed_0 = env_runtime.noise_seed_0 or 0

    env = _make_torch_env(problem)
    torch_env = env.torch_env()
    module = torch_env.module.to(device)
    module.train()

    if hasattr(module, "forward"):
        policy = module
    else:
        from problems.torch_policy import TorchPolicy

        policy = TorchPolicy(module, env_runtime)

    dim = sum(p.numel() for p in module.parameters())
    perturbator = _make_perturbator(module, num_dim_target)

    embed_module = getattr(module, "model", module)
    embed_bounds = _gym_embed_bounds(num_state)
    uhd = _make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        embed_module=embed_module,
        embed_bounds=embed_bounds,
        be=be,
    )

    frozen = bool(getattr(env_runtime, "frozen_noise", False))

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_runtime,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}, state={num_state}, action={num_action}")
    _run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
    )


def _make_gym_policy(problem, device, num_state, num_action):
    from problems.mlp_torch_policy import MLPPolicyModule
    from problems.torch_policy import TorchPolicy

    env_runtime = problem.env
    policy = problem.build_policy()
    if isinstance(policy, torch.nn.Module):
        return policy.to(device), policy
    if policy is not None:
        warnings.warn(
            f"Non-module policy {type(policy).__name__}; using MLPPolicyModule.",
            stacklevel=2,
        )
    module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
    return module, TorchPolicy(module, env_runtime)


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
    import time

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


__all__ = [
    "_count_correct",
    "_eval_full_train_acc",
    "_make_gym_policy",
    "_run_simple_gym",
    "_run_simple_gym_np",
    "_run_simple_gym_torch",
    "_run_simple_iterations",
    "_run_simple_torch",
    "_should_log_simple",
    "_try_make_np_policy",
]
