import pytest
import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator, apply_weight_decay


def _make_module():
    return nn.Linear(3, 2, bias=True)


def test_perturb_changes_params():
    module = _make_module()
    orig_weight = module.weight.data.clone()
    orig_bias = module.bias.data.clone()

    gp = GaussianPerturbator(module)
    gp.perturb(seed=42, sigma=1.0)

    assert not torch.equal(module.weight.data, orig_weight)
    assert not torch.equal(module.bias.data, orig_bias)


def test_unperturb_restores_params():
    module = _make_module()
    orig_weight = module.weight.data.clone()
    orig_bias = module.bias.data.clone()

    gp = GaussianPerturbator(module)
    gp.perturb(seed=42, sigma=1.0)
    gp.unperturb()

    assert torch.allclose(module.weight.data, orig_weight, atol=1e-6)
    assert torch.allclose(module.bias.data, orig_bias, atol=1e-6)


def test_perturb_twice_raises():
    gp = GaussianPerturbator(_make_module())
    gp.perturb(seed=0, sigma=0.1)
    with pytest.raises(AssertionError, match="Already perturbed"):
        gp.perturb(seed=1, sigma=0.1)


def test_unperturb_without_perturb_raises():
    gp = GaussianPerturbator(_make_module())
    with pytest.raises(AssertionError, match="Not perturbed"):
        gp.unperturb()


def test_accept_keeps_perturbation():
    module = _make_module()
    orig_weight = module.weight.data.clone()

    gp = GaussianPerturbator(module)
    gp.perturb(seed=42, sigma=1.0)
    perturbed_weight = module.weight.data.clone()
    gp.accept()

    # Params stay at the perturbed values
    assert torch.equal(module.weight.data, perturbed_weight)
    assert not torch.equal(module.weight.data, orig_weight)


def test_accept_without_perturb_raises():
    gp = GaussianPerturbator(_make_module())
    with pytest.raises(AssertionError, match="Not perturbed"):
        gp.accept()


def test_accept_allows_next_perturb():
    module = _make_module()
    gp = GaussianPerturbator(module)
    gp.perturb(seed=0, sigma=0.1)
    gp.accept()
    # Should not raise
    gp.perturb(seed=1, sigma=0.1)


def test_apply_weight_decay():
    module = _make_module()
    orig = module.weight.data.clone()
    apply_weight_decay(module, lr=0.01, weight_decay=0.1)
    decay = 1.0 - 0.01 * 0.1
    assert torch.allclose(module.weight.data, orig * decay)


def test_apply_weight_decay_zero_no_op():
    module = _make_module()
    orig = module.weight.data.clone()
    apply_weight_decay(module, lr=0.01, weight_decay=0.0)
    assert torch.equal(module.weight.data, orig)


def test_different_seeds_give_different_perturbations():
    m1 = _make_module()
    m2 = _make_module()
    # Same initial weights
    m2.load_state_dict(m1.state_dict())

    GaussianPerturbator(m1).perturb(seed=0, sigma=1.0)
    GaussianPerturbator(m2).perturb(seed=1, sigma=1.0)

    assert not torch.equal(m1.weight.data, m2.weight.data)
