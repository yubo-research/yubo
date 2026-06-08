import warnings

import torch

from ops.uhd_config import BEConfig
from ops.uhd_setup_monolith_opt import (
    _gym_embed_bounds,
    _make_perturbator,
    _make_simple_optimizer,
)
from ops.uhd_setup_monolith_support import (
    _action_dim,
    _evaluate_gym_with_denoise,
    _make_torch_env,
)
from ops.uhd_setup_simple_common import (
    _count_correct,
    _default_be_config,
    _eval_full_train_acc,
    _run_simple_iterations,
    _should_log_simple,
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
    from ops import uhd_setup_simple_torch as _st

    _st._run_simple_torch(
        env,
        env_runtime,
        env_tag,
        num_rounds,
        optimizer=optimizer,
        sigma=sigma,
        num_dim_target=num_dim_target,
        batch_size=batch_size,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
        num_denoise=num_denoise,
        be=be,
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
    from ops.uhd_setup_simple_common import _be_enn_kwargs
    from optimizer.uhd_mezo_np import UHDMeZOBENp, UHDMeZONp
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    noise_seed_0 = env_runtime.noise_seed_0 or 0
    frozen = bool(getattr(env_runtime, "frozen_noise", False))
    dim = policy.num_params()
    param_clip = (-1.0, 1.0)

    cfg = be if be is not None else _default_be_config()
    if optimizer in {"simple_be", "mezo_be"}:
        num_state = env_runtime.gym_conf.state_space.shape[0]
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        if optimizer == "simple_be":
            uhd = UHDSimpleBENp(
                policy,
                embedder,
                sigma_0=sigma,
                param_clip=param_clip,
                adapt_sigma=cfg.adapt_sigma,
                **_be_enn_kwargs(cfg),
            )
        else:
            uhd = UHDMeZOBENp(
                policy,
                embedder,
                sigma=sigma,
                lr=0.001,
                param_clip=param_clip,
                **_be_enn_kwargs(cfg),
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
