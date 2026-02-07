import numpy as np
import torch

from optimizer.trajectories import collect_trajectory
from problems.env_conf import default_policy, get_env_conf
from problems.mnist_classifier import MnistClassifier
from problems.mnist_env import MnistEnv, MnistEvaluator, MnistTorchEnv
from problems.torch_policy import TorchPolicy

# ── MnistEnv (param-space, for experiment.py) ──────────────────────────


def test_mnist_env_step():
    env = MnistEnv(batch_size=16)
    state, _ = env.reset(seed=42)
    assert state == 0
    num_params = env.action_space.shape[0]
    action = np.zeros(num_params, dtype=np.float32)
    state, reward, done, _ = env.step(action)
    assert state == 1
    assert done is True
    assert reward <= 0.0  # -loss
    assert reward > -100.0
    env.close()


def test_mnist_env_conf():
    ec = get_env_conf("mnist")
    assert ec.env_name == "mnist"
    assert ec.gym_conf is None
    assert ec.action_space.shape[0] > 0


def test_mnist_default_policy():
    ec = get_env_conf("mnist")
    policy = default_policy(ec)
    assert policy.num_params() == ec.action_space.shape[0]


def test_mnist_collect_trajectory():
    ec = get_env_conf("mnist")
    policy = default_policy(ec)
    traj = collect_trajectory(ec, policy, noise_seed=0)
    assert isinstance(traj.rreturn, float)
    assert traj.rreturn <= 0.0


# ── MnistEnv.torch_env() ───────────────────────────────────────────────


def test_torch_env_shares_classifier():
    env = MnistEnv(batch_size=16)
    torch_env = env.torch_env()
    assert torch_env.module is env._classifier
    env.close()


# ── MnistTorchEnv (images→logits→loss, for exp_uhd.py) ────────────────


def test_torch_env_step():
    module = MnistClassifier()
    torch_env = MnistTorchEnv(module=module, batch_size=16)
    state, _ = torch_env.reset(seed=42)
    assert state.shape == (16, 1, 28, 28)

    with torch.inference_mode():
        logits = module(torch.as_tensor(state, dtype=torch.float32))
    _, reward, done, _ = torch_env.step(logits.numpy())
    assert done is True
    assert reward <= 0.0
    assert reward > -100.0
    torch_env.close()


def test_torch_env_with_torch_policy():
    """TorchPolicy + MnistTorchEnv: the full in-place perturbation path."""

    class _FakeEnvConf:
        problem_seed = 0
        gym_conf = None

    module = MnistClassifier()
    torch_env = MnistTorchEnv(module=module, batch_size=16)
    policy = TorchPolicy(module, _FakeEnvConf())

    state, _ = torch_env.reset(seed=0)
    logits = policy(state)
    _, reward, done, _ = torch_env.step(logits)
    assert done is True
    assert isinstance(reward, float)
    assert reward <= 0.0


def test_torch_env_sees_perturbation():
    """Perturbing the module changes the logits (and reward) without param copies."""

    class _FakeEnvConf:
        problem_seed = 0
        gym_conf = None

    module = MnistClassifier()
    torch_env = MnistTorchEnv(module=module, batch_size=16)
    policy = TorchPolicy(module, _FakeEnvConf())

    # Evaluate before perturbation
    state, _ = torch_env.reset(seed=42)
    logits_before = policy(state).copy()

    # Perturb in-place
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Evaluate after — same images, different logits
    logits_after = policy(state)
    assert not np.allclose(logits_before, logits_after, atol=1e-4)


def test_torch_env_lazy_module():
    torch_env = MnistTorchEnv(batch_size=16)
    # Module is created lazily on first access
    assert isinstance(torch_env.module, MnistClassifier)


# ── MnistEvaluator (legacy, still available) ───────────────────────────


def test_evaluator_returns_negative_loss():
    module = MnistClassifier()
    evaluator = MnistEvaluator(module, batch_size=16)
    y = evaluator()
    assert y <= 0.0


def test_evaluator_returns_finite():
    module = MnistClassifier()
    evaluator = MnistEvaluator(module, batch_size=16)
    y = evaluator()
    assert isinstance(y, float)
    assert y > -100.0
