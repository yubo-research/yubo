import torch

from uhd.mk_turbo_config import _compute_signal_length_from_y, mk_turbo_config
from uhd.opt_turbo import UHDBOConfig
from uhd.param_accessor import make_param_accessor
from uhd.tm_sphere import TMSphere


def test_mk_turbo_config_raasp_basic():
    num_dim = 50
    num_raasp = 7
    conf = mk_turbo_config(use_tr=False, num_raasp=num_raasp)
    assert isinstance(conf, UHDBOConfig)
    assert hasattr(conf.perturber, "perturb") and hasattr(conf.perturber, "unperturb")

    base = torch.full((num_dim,), 0.3)
    accessor = make_param_accessor(base)
    orig = accessor.clone_flat()
    ys = [0.0]

    conf.perturber.perturb(accessor, ys)
    diff = (accessor.clone_flat() - orig).abs()
    num_changed = int((diff > 1e-12).sum().item())
    assert num_changed == num_raasp
    flat = accessor.clone_flat()
    assert torch.all(flat >= 0) and torch.all(flat <= 1)

    conf.perturber.unperturb(accessor)
    assert torch.allclose(accessor.clone_flat(), orig)


def test_mk_turbo_config_trust_region_bounds():
    num_dim = 40
    num_raasp = 5
    conf = mk_turbo_config(use_tr=True, num_raasp=num_raasp)

    params = torch.rand(num_dim)
    accessor = make_param_accessor(params)
    orig = accessor.clone_flat()
    ys = [0.1, 0.2, 0.4, 0.6, 0.5]

    conf.perturber.perturb(accessor, ys)
    diff = (accessor.clone_flat() - orig).abs()
    # With length in [0.1, 1.0], per-dim change from center should be <= 0.5
    assert torch.all(diff <= 0.5 + 1e-8)
    conf.perturber.unperturb(accessor)
    assert torch.allclose(accessor.clone_flat(), orig)


def test_compute_signal_length_from_y_bounds_and_trend():
    l1 = _compute_signal_length_from_y([0.0, 0.0, 0.0, 0.0])
    l2 = _compute_signal_length_from_y([0.0, 0.0, 1.0, 1.0])
    assert 0.1 <= l1 <= 1.0
    assert 0.1 <= l2 <= 1.0
    assert l2 <= l1 + 1e-8


def test_unperturb_always_restores():
    num_dim = 20
    num_raasp = 5
    conf = mk_turbo_config(use_tr=False, num_raasp=num_raasp)

    params = torch.full((num_dim,), 0.5)
    accessor = make_param_accessor(params)
    orig = accessor.clone_flat()
    ys = [0.0]

    conf.perturber.perturb(accessor, ys)
    assert not torch.allclose(accessor.clone_flat(), orig)

    conf.perturber.unperturb(accessor)
    assert torch.allclose(accessor.clone_flat(), orig)


def test_incorporate_with_alpha():
    num_dim = 20
    num_raasp = 5

    conf_alpha1 = mk_turbo_config(use_tr=False, num_raasp=num_raasp, alpha=1.0)
    conf_alpha05 = mk_turbo_config(use_tr=False, num_raasp=num_raasp, alpha=0.5)
    conf_alpha03 = mk_turbo_config(use_tr=False, num_raasp=num_raasp, alpha=0.3)

    ys = [0.0]

    acc1 = make_param_accessor(torch.full((num_dim,), 0.5))
    conf_alpha1.perturber.perturb(acc1, ys)
    perturbed1 = acc1.clone_flat()
    conf_alpha1.perturber.incorporate(acc1)
    assert torch.allclose(acc1.clone_flat(), perturbed1)

    acc05 = make_param_accessor(torch.full((num_dim,), 0.5))
    orig05 = acc05.clone_flat()
    conf_alpha05.perturber.perturb(acc05, ys)
    perturbed2 = acc05.clone_flat()
    conf_alpha05.perturber.incorporate(acc05)
    expected = orig05 + 0.5 * (perturbed2 - orig05)
    assert torch.allclose(acc05.clone_flat(), expected, atol=1e-6)

    acc03 = make_param_accessor(torch.full((num_dim,), 0.5))
    orig03 = acc03.clone_flat()
    conf_alpha03.perturber.perturb(acc03, ys)
    perturbed3 = acc03.clone_flat()
    conf_alpha03.perturber.incorporate(acc03)
    expected2 = orig03 + 0.3 * (perturbed3 - orig03)
    assert torch.allclose(acc03.clone_flat(), expected2, atol=1e-6)


def test_unperturb_always_restores_accessor():
    num_dim = 15
    num_active = 3
    num_raasp = 4
    conf = mk_turbo_config(use_tr=False, num_raasp=num_raasp)

    controller = TMSphere(num_dim, num_active, seed=123)
    accessor = make_param_accessor(controller)
    orig = accessor.clone_flat()
    ys = [0.0]

    conf.perturber.perturb(accessor, ys)
    assert not torch.allclose(accessor.clone_flat(), orig)

    conf.perturber.unperturb(accessor)
    assert torch.allclose(accessor.clone_flat(), orig)


def test_incorporate_with_alpha_accessor():
    num_dim = 15
    num_active = 3
    num_raasp = 4

    conf_alpha1 = mk_turbo_config(use_tr=False, num_raasp=num_raasp, alpha=1.0)
    conf_alpha05 = mk_turbo_config(use_tr=False, num_raasp=num_raasp, alpha=0.5)

    controller = TMSphere(num_dim, num_active, seed=123)
    accessor = make_param_accessor(controller)
    orig = accessor.clone_flat()
    ys = [0.0]

    conf_alpha1.perturber.perturb(accessor, ys)
    perturbed1 = accessor.clone_flat()
    conf_alpha1.perturber.incorporate(accessor)
    assert torch.allclose(accessor.clone_flat(), perturbed1)

    accessor.restore(orig)
    conf_alpha05.perturber.perturb(accessor, ys)
    perturbed2 = accessor.clone_flat()
    conf_alpha05.perturber.incorporate(accessor)
    expected = orig + 0.5 * (perturbed2 - orig)
    assert torch.allclose(accessor.clone_flat(), expected, atol=1e-6)
