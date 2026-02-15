import torch
from torch import nn

from optimizer.submodule_perturbator import SubmodulePerturbator


def _make_module():
    """Small model with 4 leaf modules that have parameters."""
    return nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
        nn.BatchNorm1d(2),
    )


def _flat_params(module):
    return torch.cat([p.data.reshape(-1).clone() for p in module.parameters()])


def _leaf_modules(module):
    return [m for m in module.modules() if list(m.parameters(recurse=False))]


def _count_perturbed_leaves(module, sp, seed, sigma=0.1):
    """Perturb, count how many leaf modules changed, then unperturb."""
    orig = {id(p): p.data.clone() for p in module.parameters()}
    sp.perturb(seed=seed, sigma=sigma)
    count = 0
    for leaf in _leaf_modules(module):
        changed = any((p.data - orig[id(p)]).abs().max() > 1e-10 for p in leaf.parameters(recurse=False))
        if changed:
            count += 1
    sp.unperturb()
    return count


def test_perturb_unperturb_roundtrip():
    module = _make_module()
    sp = SubmodulePerturbator(module, num_module_target=2)
    orig = {n: p.data.clone() for n, p in module.named_parameters()}

    sp.perturb(seed=42, sigma=0.1)
    sp.unperturb()

    for n, p in module.named_parameters():
        assert torch.allclose(p.data, orig[n], atol=1e-6)


def test_perturbation_is_submodule_sparse():
    """Some leaf modules perturbed, others untouched."""
    module = _make_module()
    sp = SubmodulePerturbator(module, num_module_target=1)

    n_leaves = len(_leaf_modules(module))
    perturbed = _count_perturbed_leaves(module, sp, seed=7)

    assert 1 <= perturbed < n_leaves


def test_seed_determinism():
    module = _make_module()
    sp = SubmodulePerturbator(module, num_module_target=2)
    orig = _flat_params(module)

    sp.perturb(seed=99, sigma=0.05)
    delta1 = _flat_params(module) - orig
    sp.unperturb()

    sp.perturb(seed=99, sigma=0.05)
    delta2 = _flat_params(module) - orig
    sp.unperturb()

    assert torch.allclose(delta1, delta2)


def test_different_seeds_give_different_selections():
    module = _make_module()
    sp = SubmodulePerturbator(module, num_module_target=2)
    orig = _flat_params(module)

    sp.perturb(seed=0, sigma=0.1)
    mask1 = (_flat_params(module) - orig).abs() > 1e-10
    sp.unperturb()

    sp.perturb(seed=1, sigma=0.1)
    mask2 = (_flat_params(module) - orig).abs() > 1e-10
    sp.unperturb()

    assert not torch.equal(mask1, mask2)


def test_num_module_target_controls_sparsity():
    """More modules targeted -> more params perturbed on average."""
    module = _make_module()
    counts = []
    for nmt in [1, 2, 4]:
        sp = SubmodulePerturbator(module, num_module_target=nmt)
        total = 0
        n_trials = 30
        for seed in range(n_trials):
            orig = _flat_params(module)
            sp.perturb(seed=seed, sigma=0.1)
            delta = _flat_params(module) - orig
            total += (delta.abs() > 1e-10).sum().item()
            sp.unperturb()
        counts.append(total / n_trials)

    assert counts[0] < counts[1] < counts[2]


def test_full_density_when_target_exceeds_count():
    """When num_module_target >= num leaf modules, all params perturbed."""
    module = _make_module()
    sp = SubmodulePerturbator(module, num_module_target=100)
    orig = _flat_params(module)

    sp.perturb(seed=0, sigma=0.1)
    delta = _flat_params(module) - orig
    assert (delta.abs() > 1e-10).all()
    sp.unperturb()


def test_fraction_target():
    """num_module_target < 1 is treated as a fraction of leaf modules."""
    module = _make_module()
    sp = SubmodulePerturbator(module, num_module_target=0.5)
    # 4 leaf modules, 50% -> ~2 selected per step.
    total = sum(_count_perturbed_leaves(module, sp, seed=s) for s in range(50))
    avg = total / 50
    assert 1.0 < avg < 3.5
