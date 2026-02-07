import torch

from problems.mnist_classifier import MnistClassifier


def test_mnist_classifier_output_shape():
    model = MnistClassifier()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_mnist_classifier_eval_single_sample():
    model = MnistClassifier()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)


def test_mnist_classifier_gradients():
    model = MnistClassifier()
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
