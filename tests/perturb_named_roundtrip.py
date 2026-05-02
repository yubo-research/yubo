from __future__ import annotations


def assert_named_param_perturb_roundtrip(sp, module) -> None:
    import torch

    orig = {n: p.data.clone() for n, p in module.named_parameters()}
    sp.perturb(seed=42, sigma=0.1)
    sp.unperturb()
    for n, p in module.named_parameters():
        assert torch.allclose(p.data, orig[n], atol=1e-6)
