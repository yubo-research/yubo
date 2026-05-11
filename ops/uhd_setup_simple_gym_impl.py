from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ops.uhd_config import BEConfig


def _c():
    return importlib.import_module("ops.uhd_setup_simple_common")


def _u():
    return importlib.import_module("ops.uhd_setup_util")


def _np_run():
    return importlib.import_module("ops.uhd_setup_simple_np_run")


def run_simple_loop(
    env_tag: str,
    num_rounds: int,
    sigma: float = 0.001,
    optimizer: str = "simple",
    *,
    num_dim_target: float | None = None,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    be: BEConfig | None = None,
) -> None:
    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed))
    env = env_conf.make()
    if hasattr(env, "torch_env"):
        torch_runner = importlib.import_module("ops.uhd_setup_simple_torch")
        torch_runner._run_simple_torch(
            env,
            env_conf,
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
    else:
        _run_simple_gym(
            env,
            env_conf,
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


def _run_simple_gym(
    env,
    env_conf,
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

    c = _c()
    np_run = _np_run()
    u = _u()

    env.close()
    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed) + 27)

    np_policy = c._try_make_np_policy(env_conf)
    if np_policy is not None:
        np_run._run_simple_gym_np(
            np_policy,
            env_conf,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            log_interval=log_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )
        return

    # Try make_torch_env for MLP policies to enable in-place perturbation
    torch_env_wrapper = env_conf.make_torch_env()
    if hasattr(torch_env_wrapper, "torch_env"):
        _run_simple_gym_torch(
            env_conf,
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

    import torch

    device = torch.device("cpu")
    num_state = env_conf.gym_conf.state_space.shape[0]
    num_action = u._action_dim(env_conf.action_space)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    module, policy = c._make_gym_policy(env_conf, device, num_state, num_action)
    dim = sum(p.numel() for p in module.parameters())
    perturbator = c._make_perturbator(module, num_dim_target)

    embed_module = getattr(module, "model", module)
    embed_bounds = c._gym_embed_bounds(num_state)
    uhd = c._make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        embed_module=embed_module,
        embed_bounds=embed_bounds,
        be=be,
    )

    frozen = bool(getattr(env_conf, "frozen_noise", False))

    def evaluate_fn():
        return u._evaluate_gym_with_denoise(
            env_conf,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}, state={num_state}, action={num_action}")
    c._run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
    )


def _run_simple_gym_torch(
    env_conf,
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
    import torch

    from common.seed_all import seed_all

    c = _c()
    u = _u()

    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed) + 27)

    device = torch.device("cpu")
    num_state = env_conf.gym_conf.state_space.shape[0]
    num_action = u._action_dim(env_conf.action_space)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    # Use make_torch_env to get proper torch env with shared module for in-place perturbation
    env = env_conf.make_torch_env()
    torch_env = env.torch_env()
    module = torch_env.module.to(device)
    module.train()

    # For MLPPolicy, the module is already a callable policy (returns numpy).
    # For raw modules, wrap with TorchPolicy.
    if hasattr(module, "forward"):
        # Module is a policy class like MLPPolicy - use it directly
        policy = module
    else:
        # Raw nn.Module - wrap with TorchPolicy
        from problems.torch_policy import TorchPolicy

        policy = TorchPolicy(module, env_conf)

    dim = sum(p.numel() for p in module.parameters())
    perturbator = c._make_perturbator(module, num_dim_target)

    embed_module = getattr(module, "model", module)
    embed_bounds = c._gym_embed_bounds(num_state)
    uhd = c._make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        embed_module=embed_module,
        embed_bounds=embed_bounds,
        be=be,
    )

    frozen = bool(getattr(env_conf, "frozen_noise", False))

    def evaluate_fn():
        return u._evaluate_gym_with_denoise(
            env_conf,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}, state={num_state}, action={num_action}")
    c._run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
    )
