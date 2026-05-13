from ops.uhd_setup_bszo_core import _run_bszo_iterations
from ops.uhd_setup_bszo_evaluate import (
    make_bszo_gym_evaluate_fn,
    make_bszo_mnist_evaluate_fn,
)
from ops.uhd_setup_monolith_support import (
    _make_accuracy_fn,
    _preload_mnist_train_to_device,
)
from optimizer.gaussian_perturbator import GaussianPerturbator


def run_bszo_loop(
    env_tag: str,
    num_steps: int,
    lr: float = 0.001,
    *,
    policy_tag: str | None = None,
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
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_bszo import UHDBSZO

    if policy_tag is None:
        policy_tag = "pure-function"

    import ops.uhd_setup_monolith_support as sup

    build_problem = sup._load_build_problem()
    problem = build_problem(env_tag, policy_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    env_runtime = problem.env
    ns0 = env_runtime.noise_seed_0 or 0
    if env_runtime.problem_seed is not None:
        seed_all(int(env_runtime.problem_seed))

    env = sup._make_torch_env(problem)
    if not hasattr(env, "torch_env"):
        raise ValueError(f"BSZO requires an environment with torch_env(), got: {env_tag}")

    device = sup._get_device()
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

    is_mnist = env_tag.startswith("mnist")

    if is_mnist:
        train_images, train_labels = _preload_mnist_train_to_device(device)
        accuracy_fn = _make_accuracy_fn(module, device)
        evaluate_fn = make_bszo_mnist_evaluate_fn(module, train_images, train_labels, batch_size, device, ns0)
    else:
        accuracy_fn = None
        evaluate_fn = make_bszo_gym_evaluate_fn(env_runtime, module, noise_seed_0_arg=noise_seed_0)

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


__all__ = ["_run_bszo_iterations", "run_bszo_loop"]
