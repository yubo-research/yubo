import torch
from botorch.models import SingleTaskGP


def _make_simple_gp():
    """Create a simple GP model for testing."""
    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    model.eval()
    return model


def test_stagger_thompson_sampler_2_init():
    from sampling.stagger_thompson_sampler_2 import StaggerThompsonSampler2

    model = _make_simple_gp()
    X_control = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    sts = StaggerThompsonSampler2(model, X_control, num_samples=10)
    assert sts is not None


def test_stagger_thompson_sampler_2_samples():
    from sampling.stagger_thompson_sampler_2 import StaggerThompsonSampler2

    model = _make_simple_gp()
    X_control = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    sts = StaggerThompsonSampler2(model, X_control, num_samples=10)

    samples = sts.samples()
    assert samples.shape == (10, 2)


def test_stagger_thompson_sampler_2_improve():
    from sampling.stagger_thompson_sampler_2 import StaggerThompsonSampler2

    model = _make_simple_gp()
    X_control = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    sts = StaggerThompsonSampler2(model, X_control, num_samples=10)

    # improve with 0 iterations should do nothing
    sts.improve(num_acc_rej=0)
    assert sts.samples().shape == (10, 2)


def test_stagger_thompson_sampler_2_refine():
    from sampling.stagger_thompson_sampler_2 import StaggerThompsonSampler2

    model = _make_simple_gp()
    X_control = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    sts = StaggerThompsonSampler2(model, X_control, num_samples=10)

    sts.refine(num_refinements=1)
    samples = sts.samples()
    assert samples.shape == (10, 2)
    assert torch.all(samples >= 0)
    assert torch.all(samples <= 1)
