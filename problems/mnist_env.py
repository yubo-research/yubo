import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch.utils.data import DataLoader

from problems.mnist_classifier import MnistClassifier

_MNIST_ROOT = "data/mnist"


def _mnist_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


_cached_dataset = None


def _get_mnist_dataset():
    global _cached_dataset  # noqa: PLW0603
    if _cached_dataset is None:
        from torchvision import datasets

        _cached_dataset = datasets.MNIST(
            root=_MNIST_ROOT,
            train=True,
            download=True,
            transform=_mnist_transform(),
        )
    return _cached_dataset


class MnistTorchEnv:
    """One-step env: state=images, action=logits, reward=-loss.

    Paired with TorchPolicy(MnistClassifier), perturbations flow through
    module.forward() with zero param copies.
    """

    def __init__(self, module, batch_size: int):
        self._module = module
        self._batch_size = batch_size
        self._dataset = _get_mnist_dataset()
        self._labels = None

        self._loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._iter = iter(self._loader)
        self._cached_seed = None
        self._cached_images = None
        self._cached_labels = None

        # observation = batch of images; action = batch of logits
        # Box(-1,1) makes _transform_action an identity in collect_trajectory.
        self.observation_space = Box(low=0.0, high=1.0, shape=(batch_size, 1, 28, 28), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(batch_size, 10), dtype=np.float32)

    @property
    def module(self):
        if self._module is None:
            self._module = MnistClassifier()
        return self._module

    def reset(self, seed=None):
        # Return cached batch if same seed (e.g. antithetic pair).
        if seed is not None and seed == self._cached_seed:
            self._labels = self._cached_labels
            return self._cached_images, None

        try:
            images, labels = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            images, labels = next(self._iter)

        self._labels = labels
        self._cached_seed = seed
        self._cached_images = images.numpy()
        self._cached_labels = labels
        return self._cached_images, None

    def step(self, action):
        logits = torch.as_tensor(action, dtype=torch.float32)
        with torch.inference_mode():
            loss = F.cross_entropy(logits, self._labels)
        return np.zeros(1), -float(loss), True, False, None

    def close(self):
        pass


class MnistEnv:
    """Param-space env for experiment.py (PureFunctionPolicy compatible).

    action = flat weight vector in [-1, 1]^d.  step() maps it to raw
    weights and evaluates.
    """

    num_dim = None  # set in __init__

    def __init__(self, batch_size: int):
        self._classifier = MnistClassifier()
        self._batch_size = batch_size
        self._dataset = _get_mnist_dataset()

        with torch.inference_mode():
            self._init_params = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self._classifier.parameters()])

        self._scale = 0.2
        num_params = len(self._init_params)
        MnistEnv.num_dim = num_params

        self.observation_space = Box(low=0.0, high=1.0, dtype=np.float32)
        self.action_space = Box(
            low=-np.ones(num_params, dtype=np.float32),
            high=np.ones(num_params, dtype=np.float32),
        )
        self._rng = None

    def torch_env(self):
        """Create a TorchEnv variant sharing this env's classifier."""
        return MnistTorchEnv(module=self._classifier, batch_size=self._batch_size)

    def reset(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return 0, None

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        assert action.shape == self._init_params.shape

        flat = self._init_params + self._scale * action
        print(f"MNIST_DEBUG: mean_param = {np.mean(flat):.4f} std_param = {np.std(flat):.4f} max_abs_action = {np.max(np.abs(action)):.4f}")
        with torch.inference_mode():
            idx = 0
            for p in self._classifier.parameters():
                size = p.numel()
                p.copy_(torch.from_numpy(flat[idx : idx + size].reshape(p.shape)).float())
                idx += size

            indices = self._rng.integers(len(self._dataset), size=self._batch_size)
            images = torch.stack([self._dataset[i][0] for i in indices])
            labels = torch.tensor([self._dataset[i][1] for i in indices])

            logits = self._classifier(images)
            loss = F.cross_entropy(logits, labels)

        return 1, -float(loss), True, False, None

    def close(self):
        pass


class MnistEvaluator:
    """Standalone evaluator for UHD (no set_params, module is perturbed directly)."""

    def __init__(self, module: torch.nn.Module, batch_size: int):
        self._module = module
        self._batch_size = batch_size
        self._dataset = _get_mnist_dataset()

    def __call__(self) -> float:
        indices = torch.randint(len(self._dataset), (self._batch_size,))
        images = torch.stack([self._dataset[i][0] for i in indices])
        labels = torch.tensor([self._dataset[i][1] for i in indices])

        with torch.inference_mode():
            logits = self._module(images)
            loss = F.cross_entropy(logits, labels)

        return -float(loss)
