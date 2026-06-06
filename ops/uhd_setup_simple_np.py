import importlib


def _imod(name: str, attr: str):
    return getattr(importlib.import_module(name), attr)


def _make_simple_loop_for_np_policy(
    env_conf,
    np_policy,
    optimizer: str,
    num_rounds: int,
    sigma: float,
    lr: float,
    log_interval: int,
    target_accuracy: float | None,
    num_denoise: int | None = None,
    be: object | None = None,
):
    seed_all = _imod("common.seed_all", "seed_all")
    BEConfig = _imod("ops.uhd_config", "BEConfig")
    BehavioralEmbedder = _imod("embedding.behavioral_embedder", "BehavioralEmbedder")
    common = importlib.import_module("ops.uhd_setup_simple_common")
    _gym_embed_bounds = getattr(common, "_gym_embed_bounds")
    _run_simple_iterations = getattr(common, "_run_simple_iterations")
    _be_enn_kwargs = getattr(common, "_be_enn_kwargs")
    UHDMeZOBENp = _imod("optimizer.uhd_mezo_np", "UHDMeZOBENp")
    UHDMeZONp = _imod("optimizer.uhd_mezo_np", "UHDMeZONp")
    UHDSimpleBENp = _imod("optimizer.uhd_simple_be_np", "UHDSimpleBENp")
    UHDSimpleNp = _imod("optimizer.uhd_simple_np", "UHDSimpleNp")

    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed) + 27)

    dim = np_policy.num_params()
    noise_seed_0 = env_conf.noise_seed_0 or 0
    frozen = bool(getattr(env_conf, "frozen_noise", False))

    param_clip = (-1.0, 1.0)
    if optimizer == "simple":
        uhd = UHDSimpleNp(np_policy, sigma_0=sigma, param_clip=param_clip)
    elif optimizer == "simple_be":
        num_state = env_conf.gym_conf.state_space.shape[0]
        cfg = be if be is not None else BEConfig()
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        uhd = UHDSimpleBENp(
            np_policy,
            embedder,
            sigma_0=sigma,
            param_clip=param_clip,
            adapt_sigma=cfg.adapt_sigma,
            **_be_enn_kwargs(cfg),
        )
    elif optimizer == "mezo":
        uhd = UHDMeZONp(np_policy, sigma=sigma, lr=lr, param_clip=param_clip)
    elif optimizer == "mezo_be":
        num_state = env_conf.gym_conf.state_space.shape[0]
        cfg = be if be is not None else BEConfig()
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        uhd = UHDMeZOBENp(
            np_policy,
            embedder,
            sigma=sigma,
            lr=lr,
            param_clip=param_clip,
            **_be_enn_kwargs(cfg),
        )
    else:
        raise ValueError(f"Unknown optimizer for numpy policy: {optimizer}")

    def evaluate_fn():
        _evaluate_gym_with_denoise = _imod("ops.uhd_setup_util", "_evaluate_gym_with_denoise")
        return _evaluate_gym_with_denoise(
            env_conf,
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
