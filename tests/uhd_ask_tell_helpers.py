"""Shared ask/tell test assertions for UHD optimizers."""

import torch


def assert_ask_perturbs_module(make_uhd):
    module, _, uhd = make_uhd()
    orig = module.weight.data.clone()
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def assert_tell_first_always_accepts(make_uhd):
    module, _, uhd = make_uhd()
    orig = module.weight.data.clone()

    uhd.ask()
    perturbed = module.weight.data.clone()
    uhd.tell(1.0, 0.0)

    assert torch.equal(module.weight.data, perturbed)
    assert not torch.equal(module.weight.data, orig)


def assert_tell_worse_reverts(make_uhd):
    module, _, uhd = make_uhd()

    uhd.ask()
    uhd.tell(10.0, 0.0)
    accepted_weight = module.weight.data.clone()

    uhd.ask()
    uhd.tell(5.0, 0.0)

    assert torch.allclose(module.weight.data, accepted_weight)


def assert_tell_improvement_keeps_new_params(make_uhd):
    module, _, uhd = make_uhd()

    uhd.ask()
    uhd.tell(1.0, 0.0)

    uhd.ask()
    better_weight = module.weight.data.clone()
    uhd.tell(2.0, 0.0)

    assert torch.equal(module.weight.data, better_weight)
