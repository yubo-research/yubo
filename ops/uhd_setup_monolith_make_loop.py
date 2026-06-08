import torch

from ops.uhd_config import BEConfig, EarlyRejectConfig
from ops.uhd_setup_monolith_simple_run import _try_make_np_policy
from ops.uhd_setup_monolith_support import (
    _action_dim,
    _evaluate_gym_with_denoise,
    _get_device,
    _load_build_problem,
    _make_accuracy_fn,
    _maybe_attach_enn,
    _parse_enn_cfg,
    _preload_mnist_train_to_device,
)
from ops.uhd_setup_simple_common import (
    _default_be_config,
    _gym_embed_bounds,
    _make_simple_optimizer,
)
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.lr_scheduler import ConstantLR
from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.submodule_perturbator import SubmodulePerturbator
from optimizer.uhd_driver import UHDDriver
from optimizer.uhd_mezo import UHDMeZO


def _make_uhd_perturbator(module, *, num_dim_target, num_module_target):
    if num_module_target is not None:
        return SubmodulePerturbator(module, num_module_target=num_module_target)
    if num_dim_target is not None:
        return SparseGaussianPerturbator(module, num_dim_target=num_dim_target)
    return GaussianPerturbator(module)


def _make_uhd_optimizer(
    optimizer: str,
    module,
    perturbator,
    *,
    dim: int,
    lr: float,
    sigma: float,
    embed_module=None,
    embed_bounds=None,
    be: BEConfig | None = None,
):
    if optimizer == "mezo":
        return UHDMeZO(
            perturbator,
            dim,
            lr_scheduler=ConstantLR(lr),
            sigma=sigma,
        )
    if optimizer in {"simple", "simple_be", "mezo_be"}:
        return _make_simple_optimizer(
            module,
            perturbator,
            optimizer=optimizer,
            sigma=sigma,
            dim=dim,
            lr=lr,
            embed_module=embed_module,
            embed_bounds=embed_bounds,
            be=be,
        )
    raise ValueError(f"Unknown UHD optimizer: {optimizer!r}")


def _make_np_uhd_optimizer(
    np_policy,
    env_runtime,
    *,
    optimizer: str,
    sigma: float,
    lr: float,
    be: BEConfig | None = None,
):
    from common.seed_all import seed_all
    from embedding.behavioral_embedder import BehavioralEmbedder
    from optimizer.uhd_mezo_np import UHDMeZOBENp, UHDMeZONp
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed) + 27)

    param_clip = (-1.0, 1.0)
    from ops.uhd_setup_simple_common import _be_enn_kwargs

    cfg = be if be is not None else _default_be_config()
    if optimizer == "simple":
        return UHDSimpleNp(np_policy, sigma_0=sigma, param_clip=param_clip)
    if optimizer == "simple_be":
        num_state = env_runtime.gym_conf.state_space.shape[0]
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        return UHDSimpleBENp(
            np_policy,
            embedder,
            sigma_0=sigma,
            param_clip=param_clip,
            adapt_sigma=cfg.adapt_sigma,
            **_be_enn_kwargs(cfg),
        )
    if optimizer == "mezo":
        return UHDMeZONp(np_policy, sigma=sigma, lr=lr, param_clip=param_clip)
    if optimizer == "mezo_be":
        num_state = env_runtime.gym_conf.state_space.shape[0]
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        return UHDMeZOBENp(
            np_policy,
            embedder,
            sigma=sigma,
            lr=lr,
            param_clip=param_clip,
            **_be_enn_kwargs(cfg),
        )
    raise ValueError(f"Unknown optimizer for numpy policy: {optimizer}")


def _make_driver_for_np_policy(
    env_runtime,
    np_policy,
    *,
    optimizer: str,
    num_rounds: int,
    sigma: float,
    lr: float,
    log_interval: int,
    target_accuracy: float | None,
    num_denoise: int | None = None,
    be: BEConfig | None = None,
) -> UHDDriver:
    noise_seed_0 = env_runtime.noise_seed_0 or 0
    frozen = bool(getattr(env_runtime, "frozen_noise", False))
    uhd = _make_np_uhd_optimizer(
        np_policy,
        env_runtime,
        optimizer=optimizer,
        sigma=sigma,
        lr=lr,
        be=be,
    )

    def evaluate_fn(eval_seed):
        return _evaluate_gym_with_denoise(
            env_runtime,
            np_policy,
            eval_seed=eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    class _NpModule:
        def parameters(self):
            return []

    class _NpPerturbator:
        pass

    return UHDDriver(
        _NpModule(),
        uhd,
        _NpPerturbator(),
        evaluate_fn,
        optimizer=optimizer,
        num_iterations=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
        print_summary=True,
    )


def make_loop(
    env_tag,
    num_rounds,
    lr=0.001,
    sigma=0.001,
    num_dim_target=None,
    num_module_target=None,
    *,
    optimizer: str = "mezo",
    policy_tag: str,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    enn: dict[str, object] | None = None,
    early_reject: EarlyRejectConfig | None = None,
    be: BEConfig | None = None,
):
    from common.seed_all import seed_all

    build_problem = _load_build_problem()
    problem = build_problem(env_tag, policy_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    env_runtime = problem.env
    noise_seed_0 = env_runtime.noise_seed_0 or 0

    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed))

    env = env_runtime.make()

    if hasattr(env, "torch_env"):
        device = _get_device()
        torch_env = env.torch_env()
        module = torch_env.module.to(device)
        module.train()

        train_images, train_labels = _preload_mnist_train_to_device(device)
        from ops.uhd_setup_mnist_loop_eval import make_uhd_mnist_torch_evaluate_fn

        _evaluate_fn = make_uhd_mnist_torch_evaluate_fn(
            env_tag,
            noise_seed_0,
            num_denoise,
            batch_size,
            module,
            device,
            train_images,
            train_labels,
        )

        accuracy_fn = _make_accuracy_fn(module, device)
        embed_module = module
        embed_bounds = None
    else:
        env.close()

        from problems.mlp_torch_policy import MLPPolicyModule

        device = torch.device("cpu")

        if env_runtime.problem_seed is not None:
            seed_all(int(env_runtime.problem_seed) + 27)

        env_runtime.ensure_spaces()
        num_state = env_runtime.gym_conf.state_space.shape[0]
        num_action = _action_dim(env_runtime.action_space)

        np_policy = _try_make_np_policy(problem)
        if np_policy is not None:
            return _make_driver_for_np_policy(
                env_runtime,
                np_policy,
                optimizer=optimizer,
                num_rounds=num_rounds,
                sigma=sigma,
                lr=lr,
                log_interval=log_interval,
                target_accuracy=target_accuracy,
                num_denoise=num_denoise,
                be=be,
            )

        policy = problem.build_policy()
        module = None
        if isinstance(policy, torch.nn.Module):
            module = policy.to(device)

        if module is None:
            module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
            from problems.torch_policy import TorchPolicy

            policy = TorchPolicy(module, env_runtime)

        def _evaluate_fn(eval_seed):
            return _evaluate_gym_with_denoise(
                env_runtime,
                policy,
                eval_seed=eval_seed,
                noise_seed_0=noise_seed_0,
                frozen=bool(getattr(env_runtime, "frozen_noise", False)),
                num_denoise=num_denoise,
            )

        accuracy_fn = None
        embed_module = getattr(module, "model", module)
        embed_bounds = _gym_embed_bounds(num_state)

    acc_fn = accuracy_fn if hasattr(env, "torch_env") else None
    dim = sum(p.numel() for p in module.parameters())
    perturbator = _make_uhd_perturbator(module, num_dim_target=num_dim_target, num_module_target=num_module_target)
    uhd = _make_uhd_optimizer(
        optimizer,
        module,
        perturbator,
        dim=dim,
        lr=lr,
        sigma=sigma,
        embed_module=embed_module,
        embed_bounds=embed_bounds,
        be=be,
    )
    loop = UHDDriver(
        module,
        uhd,
        perturbator,
        _evaluate_fn,
        optimizer=optimizer,
        num_iterations=num_rounds,
        accuracy_fn=acc_fn,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
        print_summary=True,
    )
    er = (
        early_reject
        if early_reject is not None
        else EarlyRejectConfig(
            tau=None,
            mode=None,
            ema_beta=None,
            warmup_pos=None,
            quantile=None,
            window=None,
        )
    )
    if er.tau is not None or er.mode is not None:
        loop.set_early_reject_advanced(
            tau=er.tau,
            mode="y_best" if er.mode is None else er.mode,
            ema_beta=0.99 if er.ema_beta is None else er.ema_beta,
            warmup_pos=200 if er.warmup_pos is None else er.warmup_pos,
            quantile=0.5 if er.quantile is None else er.quantile,
            window=200 if er.window is None else er.window,
        )
    enn_minus_impute, enn_cfg = _parse_enn_cfg(enn)
    _maybe_attach_enn(loop, module=module, env=env, enabled=enn_minus_impute, cfg=enn_cfg)
    return loop


__all__ = [
    "_make_driver_for_np_policy",
    "_make_np_uhd_optimizer",
    "_make_uhd_optimizer",
    "_make_uhd_perturbator",
    "make_loop",
]
