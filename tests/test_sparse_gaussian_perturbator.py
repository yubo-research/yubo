import torch
from torch import nn

from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator


def _make_module():
    return nn.Linear(50, 10, bias=True)  # 510 params


def test_perturb_unperturb_roundtrip():
    module = _make_module()
    sp = SparseGaussianPerturbator(module, num_dim_target=20)
    orig = {n: p.data.clone() for n, p in module.named_parameters()}

    sp.perturb(seed=42, sigma=0.1)
    sp.unperturb()

    for n, p in module.named_parameters():
        assert torch.allclose(p.data, orig[n], atol=1e-6)


def test_perturbation_is_sparse():
    module = _make_module()
    sp = SparseGaussianPerturbator(module, num_dim_target=20)
    orig = torch.cat([p.data.reshape(-1).clone() for p in module.parameters()])

    sp.perturb(seed=7, sigma=0.1)
    perturbed = torch.cat([p.data.reshape(-1) for p in module.parameters()])
    delta = perturbed - orig

    num_changed = (delta.abs() > 1e-10).sum().item()
    dim = orig.numel()  # 510
    # Expect ~20 dims perturbed; allow wide margin.
    assert num_changed < dim, "Should not perturb all dimensions"
    assert num_changed > 0, "Should perturb at least one dimension"


def test_seed_determinism():
    module = _make_module()
    sp = SparseGaussianPerturbator(module, num_dim_target=20)
    orig = torch.cat([p.data.reshape(-1).clone() for p in module.parameters()])

    sp.perturb(seed=99, sigma=0.05)
    delta1 = torch.cat([p.data.reshape(-1) for p in module.parameters()]) - orig
    sp.unperturb()

    sp.perturb(seed=99, sigma=0.05)
    delta2 = torch.cat([p.data.reshape(-1) for p in module.parameters()]) - orig
    sp.unperturb()

    assert torch.allclose(delta1, delta2)


def test_different_seeds_give_different_masks():
    module = _make_module()
    sp = SparseGaussianPerturbator(module, num_dim_target=20)
    orig = torch.cat([p.data.reshape(-1).clone() for p in module.parameters()])

    sp.perturb(seed=0, sigma=0.1)
    mask1 = (torch.cat([p.data.reshape(-1) for p in module.parameters()]) - orig).abs() > 1e-10
    sp.unperturb()

    sp.perturb(seed=1, sigma=0.1)
    mask2 = (torch.cat([p.data.reshape(-1) for p in module.parameters()]) - orig).abs() > 1e-10
    sp.unperturb()

    assert not torch.equal(mask1, mask2)


def test_num_dim_target_controls_sparsity():
    """More dims targeted → more dims perturbed on average."""
    module = _make_module()
    counts = []
    for ndt in [5, 50, 200]:
        sp = SparseGaussianPerturbator(module, num_dim_target=ndt)
        orig = torch.cat([p.data.reshape(-1).clone() for p in module.parameters()])
        total = 0
        n_trials = 20
        for seed in range(n_trials):
            sp.perturb(seed=seed, sigma=0.1)
            delta = torch.cat([p.data.reshape(-1) for p in module.parameters()]) - orig
            total += (delta.abs() > 1e-10).sum().item()
            sp.unperturb()
        counts.append(total / n_trials)

    # Monotonically increasing.
    assert counts[0] < counts[1] < counts[2]


def test_full_density_when_target_exceeds_dim():
    """When num_dim_target >= dim, all dimensions should be perturbed."""
    module = nn.Linear(3, 2, bias=False)  # 6 params
    sp = SparseGaussianPerturbator(module, num_dim_target=100)
    orig = module.weight.data.clone()

    sp.perturb(seed=0, sigma=0.1)
    delta = module.weight.data - orig
    assert (delta.abs() > 1e-10).all()
    sp.unperturb()
