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
from optimizer.uhd_loop import UHDLoop


def _make_simple_loop_for_np_policy(
    env_runtime,
    np_policy,
    optimizer: str,
    num_rounds: int,
    sigma: float,
    lr: float,
    log_interval: int,
    target_accuracy: float | None,
    num_denoise: int | None = None,
    be: BEConfig | None = None,
):
    from common.seed_all import seed_all
    from embedding.behavioral_embedder import BehavioralEmbedder
    from ops.uhd_setup_monolith_opt import _gym_embed_bounds
    from ops.uhd_setup_monolith_simple_run import _run_simple_iterations
    from optimizer.uhd_mezo_np import UHDMeZOBENp, UHDMeZONp
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed) + 27)

    dim = np_policy.num_params()
    noise_seed_0 = env_runtime.noise_seed_0 or 0
    frozen = bool(getattr(env_runtime, "frozen_noise", False))

    param_clip = (-1.0, 1.0)
    if optimizer == "simple":
        uhd = UHDSimpleNp(np_policy, sigma_0=sigma, param_clip=param_clip)
    elif optimizer == "simple_be":
        num_state = env_runtime.gym_conf.state_space.shape[0]
        cfg = be if be is not None else BEConfig()
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        uhd = UHDSimpleBENp(
            np_policy,
            embedder,
            sigma_0=sigma,
            param_clip=param_clip,
            num_candidates=cfg.num_candidates,
            warmup=cfg.warmup,
            fit_interval=cfg.fit_interval,
            enn_k=cfg.enn_k,
        )
    elif optimizer == "mezo":
        uhd = UHDMeZONp(np_policy, sigma=sigma, lr=lr, param_clip=param_clip)
    elif optimizer == "mezo_be":
        num_state = env_runtime.gym_conf.state_space.shape[0]
        cfg = be if be is not None else BEConfig()
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        uhd = UHDMeZOBENp(
            np_policy,
            embedder,
            sigma=sigma,
            lr=lr,
            param_clip=param_clip,
            num_candidates=cfg.num_candidates,
            warmup=cfg.warmup,
            fit_interval=cfg.fit_interval,
            enn_k=cfg.enn_k,
        )
    else:
        raise ValueError(f"Unknown optimizer for numpy policy: {optimizer}")

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_runtime,
            np_policy,
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

    class _DummyLoop:
        def run(self):
            pass

    return _DummyLoop()


def make_loop(
    env_tag,
    num_rounds,
    lr=0.001,
    sigma=0.001,
    num_dim_target=None,
    num_module_target=None,
    *,
    policy_tag: str | None = None,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    enn: dict[str, object] | None = None,
    early_reject: EarlyRejectConfig | None = None,
):
    from common.seed_all import seed_all

    if policy_tag is None:
        policy_tag = "pure-function"

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
            return _make_simple_loop_for_np_policy(
                env_runtime,
                np_policy,
                optimizer="mezo",
                num_rounds=num_rounds,
                sigma=sigma,
                lr=lr,
                log_interval=log_interval,
                target_accuracy=target_accuracy,
                num_denoise=num_denoise,
                be=BEConfig(
                    num_probes=10,
                    num_candidates=10,
                    warmup=20,
                    fit_interval=10,
                    enn_k=25,
                    sigma_range=None,
                ),
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


__all__ = ["_make_simple_loop_for_np_policy", "make_loop"]
