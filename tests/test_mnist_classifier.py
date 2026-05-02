import pytest
import torch

from optimizer.gaussian_perturbator import GaussianPerturbator
from problems.mnist_classifier import MnistClassifier


@pytest.mark.parametrize("batch,expected_rows", [(4, 4), (1, 1)])
def test_mnist_classifier_output_shape(batch, expected_rows):
    model = MnistClassifier()
    if batch == 1:
        model.eval()
    x = torch.randn(batch, 1, 28, 28)
    out = model(x)
    assert out.shape == (expected_rows, 10)


def test_mnist_classifier_gradients():
    model = MnistClassifier()
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_perturb_unperturb_roundtrip():
    model = MnistClassifier()
    orig = [p.data.clone() for p in model.parameters()]

    gp = GaussianPerturbator(model)
    gp.perturb(seed=42, sigma=1.0)
    gp.unperturb()

    for p, o in zip(model.parameters(), orig, strict=True):
        assert torch.allclose(p.data, o, atol=1e-6)
