import numpy as np
import torch.nn as nn

from problems.torch_policy import TorchPolicy


class _FakeGymConf:
    def __init__(self, num_state):
        self.state_space = type("S", (), {"shape": (num_state,)})()
        self.transform_state = True


class _FakeEnvConf:
    def __init__(self, num_state):
        self.problem_seed = 0
        self.gym_conf = _FakeGymConf(num_state)


class _FakeEnvConfNoGym:
    problem_seed = 0
    gym_conf = None


def test_call_returns_numpy():
    module = nn.Linear(4, 2)
    policy = TorchPolicy(module, _FakeEnvConf(4))
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)


def test_output_clamped():
    module = nn.Linear(4, 2)
    # Set large weights to force large outputs
    nn.init.constant_(module.weight, 100.0)
    nn.init.constant_(module.bias, 100.0)
    policy = TorchPolicy(module, _FakeEnvConf(4))
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action.max() <= 1.0
    assert action.min() >= -1.0


def test_no_gym_conf_skips_normalization_and_clamping():
    module = nn.Linear(4, 2)
    nn.init.constant_(module.weight, 100.0)
    nn.init.constant_(module.bias, 100.0)
    policy = TorchPolicy(module, _FakeEnvConfNoGym())

    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    # Without clamping, large weights produce outputs >> 1
    assert action.max() > 1.0


def test_no_gym_conf_handles_multidim_state():
    """TorchPolicy with gym_conf=None should handle non-1D states (e.g. images)."""
    # Simple conv module that takes (batch, 1, 4, 4) input
    module = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))
    policy = TorchPolicy(module, _FakeEnvConfNoGym())

    state = np.random.rand(2, 1, 4, 4).astype(np.float32)
    action = policy(state)
    assert action.shape == (2, 3)


def test_gaussian_policy_stochastic_shape_and_bounds():
    from problems.torch_policy import GaussianTorchPolicy
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        4,
        2,
        BackboneSpec("mlp", (8,), "relu", layer_norm=False),
        HeadSpec(),
    )
    policy = GaussianTorchPolicy(module, _FakeEnvConf(4), deterministic_eval=False, squash_mode="clip")
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action.shape == (2,)
    assert action.min() >= -1.0
    assert action.max() <= 1.0


def test_gaussian_policy_deterministic_eval_uses_mean_with_clip():
    import torch

    from problems.torch_policy import GaussianTorchPolicy
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        4,
        2,
        BackboneSpec("mlp", (), "tanh", layer_norm=False),
        HeadSpec(),
    )
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        module.head.bias.copy_(torch.tensor([2.0, -2.0]))
    policy = GaussianTorchPolicy(module, _FakeEnvConf(4), deterministic_eval=True, squash_mode="clip")
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    np.testing.assert_allclose(action, np.array([1.0, -1.0], dtype=np.float32), atol=1e-6)


def test_gaussian_policy_tanh_clip_mode():
    import torch

    from problems.torch_policy import GaussianTorchPolicy
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        4,
        2,
        BackboneSpec("mlp", (), "tanh", layer_norm=False),
        HeadSpec(),
    )
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        module.head.bias.copy_(torch.tensor([4.0, -4.0]))
    policy = GaussianTorchPolicy(module, _FakeEnvConf(4), deterministic_eval=True, squash_mode="tanh_clip")
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action[0] > 0.99
    assert action[1] < -0.99


def test_gaussian_policy_sample_with_log_prob_batch():
    from problems.torch_policy import GaussianTorchPolicy
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        4,
        2,
        BackboneSpec("mlp", (8,), "relu", layer_norm=False),
        HeadSpec(),
    )
    policy = GaussianTorchPolicy(module, _FakeEnvConf(4), deterministic_eval=False, squash_mode="clip")
    state = np.ones((3, 4), dtype=np.float32)
    action, log_prob = policy.sample_with_log_prob(state, deterministic=False)
    assert action.shape == (3, 2)
    assert log_prob.shape == (3,)
