#!/usr/bin/env python

import math
import warnings

import click
import torch
import torch.nn.functional as F

# Ensure Atari/DM-Control support is available when exp_uhd is used with those env tags.
import problems.env_conf_atari_dm  # noqa: F401
from optimizer.uhd_loop import UHDLoop


@click.group()
def cli():
    pass


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_accuracy_fn(module, device):
    from torchvision import datasets

    from problems.mnist_env import _MNIST_ROOT, _mnist_transform

    test_dataset = datasets.MNIST(
        root=_MNIST_ROOT,
        train=False,
        download=True,
        transform=_mnist_transform(),
    )
    images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    def accuracy_fn():
        module.eval()
        with torch.no_grad():
            preds = module(images).argmax(dim=1)
        module.train()
        return float((preds == labels).float().mean())

    return accuracy_fn


def _parse_perturb(perturb: str) -> tuple[float | None, float | None]:
    """Parse --perturb flag into (num_dim_target, num_module_target)."""
    if perturb == "dense":
        return None, None
    if perturb.startswith("dim:"):
        return float(perturb[4:]), None
    if perturb.startswith("mod:"):
        return None, float(perturb[4:])
    msg = f"Invalid --perturb value: {perturb!r}. Use 'dense', 'dim:<n>', or 'mod:<n>'."
    raise click.BadParameter(msg)


def _make_loop(
    env_tag,
    num_rounds,
    lr=0.001,
    sigma=0.001,
    num_dim_target=None,
    num_module_target=None,
):
    from problems.env_conf import get_env_conf
    from problems.torch_policy import TorchPolicy

    device = _get_device()
    env_conf = get_env_conf(env_tag)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    env = env_conf.make()

    if hasattr(env, "torch_env"):
        # Env provides a TorchEnv (e.g. MNIST).
        # TorchPolicy runs module.forward() on data from the env;
        # GaussianPerturbator perturbs the same module in-place.
        torch_env = env.torch_env()
        module = torch_env.module.to(device)
        module.train()  # BN uses batch statistics, matching fit_mnist.py
        policy = TorchPolicy(module, env_conf)

        def evaluate_fn(eval_seed):
            noise_seed = eval_seed + noise_seed_0
            state, _ = torch_env.reset(seed=noise_seed)
            logits = policy(state)
            logits_t = torch.as_tensor(logits, dtype=torch.float32)
            with torch.inference_mode():
                per_sample = F.cross_entropy(logits_t, torch_env._labels, reduction="none")
            mu = -float(per_sample.mean())
            se = float(per_sample.std() / math.sqrt(len(per_sample)))
            return mu, se

        accuracy_fn = _make_accuracy_fn(module, device)
    else:
        # Gym env — build policy network, evaluate via collect_trajectory.
        env.close()

        from optimizer.trajectories import collect_trajectory
        from problems.mlp_torch_policy import MLPPolicyModule

        if env_conf.policy_class is not None:
            warnings.warn(
                f"Replacing policy_class {env_conf.policy_class} with MLPPolicyModule",
                stacklevel=2,
            )

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]

        module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
        policy = TorchPolicy(module, env_conf)

        def evaluate_fn(eval_seed):
            noise_seed = eval_seed + noise_seed_0
            return float(collect_trajectory(env_conf, policy, noise_seed=noise_seed).rreturn), 0.0

    acc_fn = accuracy_fn if hasattr(env, "torch_env") else None
    return UHDLoop(
        module,
        evaluate_fn,
        num_iterations=num_rounds,
        lr=lr,
        sigma=sigma,
        accuracy_fn=acc_fn,
        num_dim_target=num_dim_target,
        num_module_target=num_module_target,
    )


@cli.command()
@click.option(
    "--env-tag",
    required=True,
    help="Environment tag (e.g. lunar, ant, bw, mnist)",
)
@click.option("--num-rounds", required=True, type=int, help="Number of UHD iterations")
@click.option("--lr", default=0.001, type=float, help="Max learning rate")
@click.option(
    "--perturb",
    default="dim:0.5",
    help="Perturbation strategy: 'dense', 'dim:<target>', or 'mod:<target>'",
)
def local(env_tag, num_rounds, lr, perturb):
    ndt, nmt = _parse_perturb(perturb)
    loop = _make_loop(
        env_tag,
        num_rounds,
        lr=lr,
        sigma=0.001,
        num_dim_target=ndt,
        num_module_target=nmt,
    )
    loop.run()


if __name__ == "__main__":
    cli()
