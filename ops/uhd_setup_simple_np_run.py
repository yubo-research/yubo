import importlib


def _imod(name: str, attr: str):
    return getattr(importlib.import_module(name), attr)


def _run_simple_gym_np(
    policy,
    env_conf,
    num_rounds,
    *,
    optimizer,
    sigma,
    log_interval,
    target_accuracy,
    num_denoise,
    be: object | None = None,
):
    BehavioralEmbedder = _imod("embedding.behavioral_embedder", "BehavioralEmbedder")
    common = importlib.import_module("ops.uhd_setup_simple_common")
    _default_be_config = getattr(common, "_default_be_config")
    _gym_embed_bounds = getattr(common, "_gym_embed_bounds")
    _run_simple_iterations = getattr(common, "_run_simple_iterations")
    UHDMeZOBENp = _imod("optimizer.uhd_mezo_np", "UHDMeZOBENp")
    UHDMeZONp = _imod("optimizer.uhd_mezo_np", "UHDMeZONp")
    UHDSimpleBENp = _imod("optimizer.uhd_simple_be_np", "UHDSimpleBENp")
    UHDSimpleNp = _imod("optimizer.uhd_simple_np", "UHDSimpleNp")

    noise_seed_0 = env_conf.noise_seed_0 or 0
    frozen = bool(getattr(env_conf, "frozen_noise", False))
    dim = policy.num_params()
    param_clip = (-1.0, 1.0)

    cfg = be if be is not None else _default_be_config()
    if optimizer in {"simple_be", "mezo_be"}:
        num_state = env_conf.gym_conf.state_space.shape[0]
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
        _evaluate_gym_with_denoise = _imod("ops.uhd_setup_util", "_evaluate_gym_with_denoise")
        return _evaluate_gym_with_denoise(
            env_conf,
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
