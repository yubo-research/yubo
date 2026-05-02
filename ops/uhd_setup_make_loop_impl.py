from __future__ import annotations


def make_loop(
    env_tag,
    num_rounds,
    lr=0.001,
    sigma=0.001,
    num_dim_target=None,
    num_module_target=None,
    *,
    policy_tag=None,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    enn: dict[str, object] | None = None,
    early_reject=None,
):
    import torch

    from common.seed_all import seed_all
    from ops.uhd_config import EarlyRejectConfig as _ERC
    from ops.uhd_setup_simple_common import _try_make_np_policy
    from ops.uhd_setup_simple_np import _make_simple_loop_for_np_policy
    from ops.uhd_setup_util import (
        _action_dim,
        _evaluate_gym_with_denoise,
        _get_device,
        _make_accuracy_fn,
        _maybe_attach_enn,
        _parse_enn_cfg,
        _preload_mnist_train_to_device,
    )
    from optimizer.uhd_loop import UHDLoop
    from problems.env_conf import get_env_conf
    from problems.torch_policy import TorchPolicy

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed))

    env = env_conf.make()

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
    else:
        env.close()

        from problems.mlp_torch_policy import MLPPolicyModule

        device = torch.device("cpu")

        if env_conf.problem_seed is not None:
            seed_all(int(env_conf.problem_seed) + 27)

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = _action_dim(env_conf.action_space)

        np_policy = _try_make_np_policy(env_conf)
        if np_policy is not None:
            return _make_simple_loop_for_np_policy(
                env_conf,
                np_policy,
                optimizer="mezo",
                num_rounds=num_rounds,
                sigma=sigma,
                lr=lr,
                log_interval=log_interval,
                target_accuracy=target_accuracy,
                num_denoise=num_denoise,
            )

        # Torch-based policy for continuous action spaces
        policy = None
        module = None
        if env_conf.policy_class is not None:
            cand = env_conf.policy_class(env_conf)
            if isinstance(cand, torch.nn.Module):
                module = cand.to(device)
                policy = cand

        if policy is None:
            module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
            from problems.torch_policy import TorchPolicy

            policy = TorchPolicy(module, env_conf)

        def _evaluate_fn(eval_seed):
            return _evaluate_gym_with_denoise(
                env_conf,
                policy,
                eval_seed=eval_seed,
                noise_seed_0=noise_seed_0,
                frozen=bool(getattr(env_conf, "frozen_noise", False)),
                num_denoise=num_denoise,
            )

        accuracy_fn = None

    acc_fn = accuracy_fn if hasattr(env, "torch_env") else None
    loop = UHDLoop(
        module,
        _evaluate_fn,
        num_iterations=num_rounds,
        lr=lr,
        sigma=sigma,
        accuracy_fn=acc_fn,
        num_dim_target=num_dim_target,
        num_module_target=num_module_target,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
        print_summary=True,
    )
    er = (
        early_reject
        if early_reject is not None
        else _ERC(
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
