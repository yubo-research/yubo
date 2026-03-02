import numpy as np
import torch

from problems.mlp_torch_policy import MLPPolicyModule
from problems.torch_policy import TorchPolicy


class _FakeGymConf:
    def __init__(self, num_state):
        self.state_space = type("S", (), {"shape": (num_state,)})()
        self.transform_state = True


class _FakeEnvConf:
    def __init__(self, num_state):
        self.problem_seed = 0
        self.gym_conf = _FakeGymConf(num_state)


def test_forward_returns_correct_shape():
    module = MLPPolicyModule(8, 3)
    x = torch.randn(8)
    y = module(x)
    assert y.shape == (3,)


def test_custom_hidden_sizes():
    module = MLPPolicyModule(4, 2, hidden_sizes=(64, 32, 16))
    x = torch.randn(4)
    y = module(x)
    assert y.shape == (2,)


def test_works_with_torch_policy():
    module = MLPPolicyModule(4, 2)
    policy = TorchPolicy(module, _FakeEnvConf(4))
    state = np.ones(4, dtype=np.float32)
    action = policy(state)
    assert action.shape == (2,)
    assert action.min() >= -1.0
    assert action.max() <= 1.0


def test_gaussian_actor_shapes_single():
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        5,
        3,
        BackboneSpec("mlp", (8,), "tanh", layer_norm=False),
        HeadSpec(),
    )
    x = torch.randn(5)
    mean = module(x)
    assert mean.shape == (3,)

    dist = module.dist(x)
    assert dist.mean.shape == (3,)
    assert dist.scale.shape == (3,)

    action, log_prob, entropy = module.sample_action(x, deterministic=False)
    assert action.shape == (3,)
    assert log_prob.shape == ()
    assert entropy.shape == ()


def test_gaussian_actor_shapes_batch():
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        4,
        2,
        BackboneSpec("mlp", (16, 8), "relu", layer_norm=False),
        HeadSpec(),
    )
    x = torch.randn(7, 4)
    action, log_prob, entropy = module.sample_action(x, deterministic=False)
    assert action.shape == (7, 2)
    assert log_prob.shape == (7,)
    assert entropy.shape == (7,)


def test_gaussian_actor_log_std_state_independent():
    from rl.backbone import BackboneSpec, HeadSpec
    from rl.shared_gaussian_actor import SharedGaussianActorModule

    module = SharedGaussianActorModule(
        4,
        2,
        BackboneSpec("mlp", (8,), "silu", layer_norm=False),
        HeadSpec(),
    )
    with torch.no_grad():
        module.log_std.copy_(torch.tensor([0.1, -0.3]))
    x = torch.randn(5, 4)
    dist = module.dist(x)
    for i in range(1, 5):
        assert torch.allclose(dist.scale[0], dist.scale[i])
