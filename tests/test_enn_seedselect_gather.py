import torch
from torch import nn

from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.uhd_enn_seed_selector import ENNMuPlusSeedSelector, ENNSeedSelectConfig


def _clone_params(module: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def test_enn_muplus_seed_selector_gather_does_not_leave_module_perturbed():
    module = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))
    perturbator = SparseGaussianPerturbator(module, num_dim_target=0.5)
    selector = ENNMuPlusSeedSelector(
        module=module,
        perturbator=perturbator,
        cfg=ENNSeedSelectConfig(
            d=32,
            num_candidates=2,
            warmup_real_obs=10_000,  # never fit; just exercise embedding path
            embedder="gather",
            gather_t=16,
        ),
    )

    params0 = _clone_params(module)
    chosen, ucb = selector.choose_seed_ucb(base_seed=123, sigma=0.01)
    assert chosen == 123
    assert ucb is None
    params1 = _clone_params(module)
    for a, b in zip(params0, params1, strict=True):
        torch.testing.assert_close(a, b, rtol=0.0, atol=1e-6)

    selector.tell_mu_plus(mu_plus=0.5)
    assert len(selector._x) == 1
    assert selector._x[0].shape == (32,)
    assert len(selector._y) == 1

    params2 = _clone_params(module)
    for a, b in zip(params0, params2, strict=True):
        torch.testing.assert_close(a, b, rtol=0.0, atol=1e-6)
