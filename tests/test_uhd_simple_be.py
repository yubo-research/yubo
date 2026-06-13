import torch
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_simple_be import UHDSimpleBE


def _make_uhd_be(
    sigma_0=1.0,
    warmup=5,
    num_candidates=3,
    fit_interval=1,
    sigma_range=None,
    adapt_sigma=True,
):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bounds = torch.zeros(2, 3)
    bounds[1] = 1.0
    embedder = BehavioralEmbedder(bounds, num_probes=4, seed=0)
    uhd = UHDSimpleBE(
        gp,
        sigma_0=sigma_0,
        dim=dim,
        module=module,
        embedder=embedder,
        num_candidates=num_candidates,
        warmup=warmup,
        fit_interval=fit_interval,
        sigma_range=sigma_range,
        adapt_sigma=adapt_sigma,
    )
    return module, uhd


def test_ask_perturbs_module():
    module, uhd = _make_uhd_be()
    orig = module.weight.data.clone()
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_tell_first_always_accepts():
    module, uhd = _make_uhd_be()
    uhd.ask()
    perturbed = module.weight.data.clone()
    uhd.tell(1.0, 0.0)
    assert torch.equal(module.weight.data, perturbed)


def test_tell_worse_reverts():
    module, uhd = _make_uhd_be()
    uhd.ask()
    uhd.tell(10.0, 0.0)
    accepted = module.weight.data.clone()

    uhd.ask()
    uhd.tell(5.0, 0.0)
    assert torch.allclose(module.weight.data, accepted)


def test_y_best_tracks_maximum():
    _, uhd = _make_uhd_be()
    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert uhd.y_best == 1.0

    uhd.ask()
    uhd.tell(3.0, 0.0)
    assert uhd.y_best == 3.0

    uhd.ask()
    uhd.tell(2.0, 0.0)
    assert uhd.y_best == 3.0


def test_properties():
    _, uhd = _make_uhd_be()
    assert uhd.y_best is None
    uhd.ask()
    uhd.tell(1.5, 0.3)
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.3
    assert isinstance(uhd.eval_seed, int)
    assert isinstance(uhd.sigma, float)


def test_warmup_seeds_sequential():
    _, uhd = _make_uhd_be(warmup=100)
    seeds = []
    for _ in range(5):
        uhd.ask()
        seeds.append(uhd.eval_seed)
        uhd.tell(float(torch.randn(1).item()), 0.0)
    assert seeds == [0, 1, 2, 3, 4]


def test_after_warmup_seed_in_candidate_batch_and_stream_continues():
    num_candidates = 3
    warmup = 3
    _, uhd = _make_uhd_be(warmup=warmup, num_candidates=num_candidates)

    for i in range(warmup):
        uhd.ask()
        uhd.tell(float(i), 0.0)

    uhd.ask()
    seed_after_warmup = uhd.eval_seed
    assert warmup <= seed_after_warmup < warmup + num_candidates

    uhd.tell(1.0, 0.0)
    uhd.ask()
    assert uhd.eval_seed == seed_after_warmup + 1


def test_runs_many_steps():
    from tests.uhd_linear_embed_parts import run_many_random_ask_tell

    module, uhd = _make_uhd_be(warmup=5, num_candidates=3, fit_interval=2)
    run_many_random_ask_tell(uhd, module, n=20)


def test_sample_sigmas_log_uniform():
    import numpy as np

    _, uhd = _make_uhd_be(sigma_range=(1e-5, 1e-1), num_candidates=50)
    sigmas = uhd._sample_sigmas(base_seed=42, n=50)
    assert sigmas.shape == (50,)
    assert np.all(sigmas >= 1e-5)
    assert np.all(sigmas <= 1e-1)
    assert not np.allclose(sigmas, sigmas[0])


def test_sample_sigmas_none_uses_adapter():
    import numpy as np

    _, uhd = _make_uhd_be(sigma_0=0.5, num_candidates=5)
    sigmas = uhd._sample_sigmas(base_seed=0, n=5)
    assert np.allclose(sigmas, 0.5)


def test_adapt_sigma_false_keeps_sigma_on_reject():
    _, uhd = _make_uhd_be(sigma_0=0.5, adapt_sigma=False)
    uhd.ask()
    uhd.tell(10.0, 0.0)
    sigma_after_accept = uhd.sigma

    uhd.ask()
    uhd.tell(5.0, 0.0)
    assert uhd.sigma == sigma_after_accept


def test_enn_reject_can_halve_sigma_after_tolerance():
    _, uhd = _make_uhd_be(warmup=2, num_candidates=3, sigma_0=0.5, fit_interval=10)
    for i in range(2):
        uhd.ask()
        uhd.tell(float(i), 0.0)
    uhd.ask()
    uhd.tell(1.0, 0.0)
    sigma0 = uhd.sigma
    for _ in range(20):
        uhd.ask()
        uhd.tell(0.0, 0.0)
    assert uhd.sigma < sigma0


def test_sequential_reject_still_adapts_sigma():
    _, uhd = _make_uhd_be(warmup=100, sigma_0=0.5)
    uhd.ask()
    uhd.tell(10.0, 0.0)
    sigma_after_accept = uhd.sigma

    for _ in range(20):
        uhd.ask()
        uhd.tell(0.0, 0.0)

    assert uhd.sigma < sigma_after_accept


def test_select_seed_ignores_sigma_range():
    _, uhd = _make_uhd_be(warmup=1, sigma_range=(1e-4, 1e-1), num_candidates=20, sigma_0=0.05)
    uhd.ask()
    uhd.tell(1.0, 0.0)

    called = []
    orig = uhd._sample_sigmas

    def track(*args, **kwargs):
        called.append(True)
        return orig(*args, **kwargs)

    uhd._sample_sigmas = track
    uhd.ask()
    assert called == []


def test_runs_many_steps_with_sigma_range():
    module, uhd = _make_uhd_be(warmup=5, num_candidates=3, fit_interval=2, sigma_range=(1e-5, 1e-1))
    for _ in range(20):
        uhd.ask()
        mu = float(torch.randn(1).item())
        uhd.tell(mu, 0.0)
    assert torch.isfinite(module.weight.data).all()
