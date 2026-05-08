from __future__ import annotations


def run_bszo_loop(
    env_tag: str,
    num_steps: int,
    lr: float = 0.001,
    *,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    bszo_k: int = 2,
    bszo_epsilon: float = 1e-4,
    bszo_sigma_p_sq: float = 1.0,
    bszo_sigma_e_sq: float = 1.0,
    bszo_alpha: float = 0.1,
) -> None:
    from common.seed_all import seed_all
    from ops.uhd_setup_bszo_evaluate import make_bszo_gym_evaluate_fn, make_bszo_mnist_evaluate_fn
    from ops.uhd_setup_util import _get_device, _make_accuracy_fn, _preload_mnist_train_to_device
    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_bszo import UHDBSZO
    from problems.env_conf import get_env_conf

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    ns0 = env_conf.noise_seed_0 or 0
    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed))

    # Try to make torch env (for MLP policies or MNIST-style envs)
    env = env_conf.make_torch_env()
    if not hasattr(env, "torch_env"):
        raise ValueError(f"BSZO requires an environment with torch_env(), got: {env_tag}")

    device = _get_device()
    module = env.torch_env().module.to(device)
    module.train()

    dim = sum(p.numel() for p in module.parameters())
    perturbator = GaussianPerturbator(module)
    bszo = UHDBSZO(
        perturbator,
        dim,
        lr_scheduler=ConstantLR(lr),
        epsilon=bszo_epsilon,
        k=bszo_k,
        sigma_p_sq=bszo_sigma_p_sq,
        sigma_e_sq=bszo_sigma_e_sq,
        alpha=bszo_alpha,
    )

    # Check if this is an MNIST environment (has MNIST data)
    is_mnist = env_tag.startswith("mnist")

    if is_mnist:
        train_images, train_labels = _preload_mnist_train_to_device(device)
        accuracy_fn = _make_accuracy_fn(module, device)
        evaluate_fn = make_bszo_mnist_evaluate_fn(module, train_images, train_labels, batch_size, device, ns0)
    else:
        accuracy_fn = None
        evaluate_fn = make_bszo_gym_evaluate_fn(env_conf, module, noise_seed_0_arg=noise_seed_0)

    print(f"BSZO: num_params = {dim}, k = {bszo_k}, epsilon = {bszo_epsilon}, lr = {lr}")
    _run_bszo_iterations(
        bszo,
        evaluate_fn=evaluate_fn,
        accuracy_fn=accuracy_fn,
        num_steps=num_steps,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
    )


def _run_bszo_iterations(
    bszo,
    *,
    evaluate_fn,
    accuracy_fn,
    num_steps: int,
    log_interval: int,
    accuracy_interval: int,
    target_accuracy: float | None,
) -> None:
    import time

    from ops.uhd_setup_simple_common import _should_log_simple

    k = bszo.k
    t0 = time.perf_counter()
    acc = None
    step = 0
    for step in range(num_steps):
        for _ in range(k + 1):
            bszo.ask()
            mu, se = evaluate_fn(bszo.eval_seed)
            bszo.tell(mu, se)

        if not _should_log_simple(step, num_steps, log_interval):
            continue
        y_best = bszo.y_best
        y_str = f"{y_best:.4f}" if y_best is not None else "N/A"
        if accuracy_fn is not None and (acc is None or step == num_steps - 1 or (accuracy_interval > 0 and step % accuracy_interval == 0)):
            acc = accuracy_fn()
        line = f"EVAL: step = {step} mu = {mu:.4f} se = {se:.4f} y_best = {y_str}"
        if acc is not None:
            line += f" test_acc = {acc:.4f}"
        print(line)
        if target_accuracy is not None and acc is not None and acc >= target_accuracy:
            elapsed = time.perf_counter() - t0
            print(f"BSZO: target reached {acc:.4f} >= {target_accuracy:.4f} at step={step} ({elapsed:.2f}s)")
            break

    elapsed = time.perf_counter() - t0
    print(f"BSZO: elapsed = {elapsed:.2f}s ({min(step + 1, num_steps)} steps)")
