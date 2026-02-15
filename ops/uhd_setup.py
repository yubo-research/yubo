import math
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from optimizer.uhd_loop import UHDLoop


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


def _preload_mnist_train_to_device(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load full MNIST train set as tensors and move once to device.

    This removes per-eval DataLoader overhead and repeated host→device copies.
    """
    from torchvision import datasets

    from problems.mnist_env import _MNIST_ROOT, _mnist_transform

    train_dataset = datasets.MNIST(
        root=_MNIST_ROOT,
        train=True,
        download=True,
        transform=_mnist_transform(),
    )
    loader = DataLoader(train_dataset, batch_size=8192, shuffle=False, drop_last=False)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    images = torch.cat(xs, dim=0).to(device)
    labels = torch.cat(ys, dim=0).to(device)
    return images, labels


def make_loop(
    env_tag,
    num_rounds,
    lr=0.001,
    sigma=0.001,
    num_dim_target=None,
    num_module_target=None,
    *,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
):
    from problems.env_conf import get_env_conf
    from problems.torch_policy import TorchPolicy

    device = _get_device()
    env_conf = get_env_conf(env_tag)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    env = env_conf.make()

    if hasattr(env, "torch_env"):
        torch_env = env.torch_env()
        module = torch_env.module.to(device)
        module.train()

        # MNIST-specific fast path: preload full train tensors to device and sample batches by seed.
        # This preserves antithetic pairing (same eval_seed → same sampled batch).
        train_images, train_labels = _preload_mnist_train_to_device(device)
        batch_size = int(getattr(torch_env, "_batch_size", 4096))

        def evaluate_fn(eval_seed):
            noise_seed = eval_seed + noise_seed_0
            g = torch.Generator()
            g.manual_seed(int(noise_seed))
            idx = torch.randint(train_images.shape[0], (batch_size,), generator=g)
            idx_d = idx.to(device)
            images = train_images.index_select(0, idx_d)
            labels = train_labels.index_select(0, idx_d)
            with torch.inference_mode():
                logits = module(images)
                per_sample = F.cross_entropy(logits, labels, reduction="none")
            mu = -float(per_sample.mean())
            se = float(per_sample.std() / math.sqrt(len(per_sample)))
            return mu, se

        accuracy_fn = _make_accuracy_fn(module, device)
    else:
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
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
        print_summary=True,
    )
