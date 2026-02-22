import torch
import torch.nn as nn

from embedding.behavioral_embedder import BehavioralEmbedder


def _make_linear_module():
    module = nn.Linear(3, 2, bias=False)
    module.weight.data.fill_(1.0)
    return module


def test_output_length():
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    be = BehavioralEmbedder(bounds, num_probes=5, seed=0)
    result = be.embed(_make_linear_module())
    assert result.shape == (5 * 2,)


def test_deterministic_with_same_seed():
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    be1 = BehavioralEmbedder(bounds, num_probes=4, seed=42)
    be2 = BehavioralEmbedder(bounds, num_probes=4, seed=42)
    module = _make_linear_module()
    assert torch.equal(be1.embed(module), be2.embed(module))


def test_different_seeds_differ():
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    be1 = BehavioralEmbedder(bounds, num_probes=4, seed=0)
    be2 = BehavioralEmbedder(bounds, num_probes=4, seed=1)
    module = _make_linear_module()
    assert not torch.equal(be1.embed(module), be2.embed(module))


def test_probes_within_bounds():
    lb = torch.tensor([-2.0, 0.0, 5.0])
    ub = torch.tensor([2.0, 1.0, 10.0])
    bounds = torch.stack([lb, ub])
    be = BehavioralEmbedder(bounds, num_probes=100, seed=0)
    assert (be.probes >= lb).all()
    assert (be.probes <= ub).all()


def test_different_modules_differ():
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    be = BehavioralEmbedder(bounds, num_probes=4, seed=0)

    m1 = nn.Linear(3, 2, bias=False)
    m1.weight.data.fill_(1.0)
    m2 = nn.Linear(3, 2, bias=False)
    m2.weight.data.fill_(2.0)

    assert not torch.equal(be.embed(m1), be.embed(m2))


def test_multidim_bounds():
    bounds = torch.zeros(2, 1, 4, 4)
    bounds[1] = 1.0
    be = BehavioralEmbedder(bounds, num_probes=3, seed=0)
    assert be.probes.shape == (3, 1, 4, 4)

    module = nn.Sequential(nn.Flatten(), nn.Linear(16, 5))
    result = be.embed(module)
    assert result.shape == (3 * 5,)


def test_embed_values_match_manual_forward():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    be = BehavioralEmbedder(bounds, num_probes=3, seed=7)
    module = nn.Linear(2, 4, bias=False)
    module.weight.data.fill_(0.5)

    result = be.embed(module)

    with torch.inference_mode():
        expected = module(be.probes).reshape(-1)
    assert torch.equal(result, expected)
