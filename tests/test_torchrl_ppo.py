import rl.algos.torchrl_on_policy_core as torchrl_on_policy_core
from rl.algos import torchrl_ppo
from rl.algos.torchrl_on_policy_core import _TanhNormal as _CoreTanhNormal


def test_ppo_tanhnormal_support_property():
    dist = torchrl_ppo._TanhNormal(
        loc=torchrl_ppo.torch.zeros(1),
        scale=torchrl_ppo.torch.ones(1),
    )
    assert dist.support is torchrl_ppo.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_property():
    dist = _CoreTanhNormal(
        loc=torchrl_ppo.torch.zeros(1),
        scale=torchrl_ppo.torch.ones(1),
    )
    assert dist.support is torchrl_ppo.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_property_module_path():
    dist = torchrl_on_policy_core._TanhNormal(
        loc=torchrl_ppo.torch.zeros(1),
        scale=torchrl_ppo.torch.ones(1),
    )
    assert dist.support is torchrl_ppo.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_fget_direct_call():
    dist = torchrl_on_policy_core._TanhNormal(
        loc=torchrl_ppo.torch.zeros(1),
        scale=torchrl_ppo.torch.ones(1),
    )
    support_value = torchrl_on_policy_core._TanhNormal.support.fget(dist)
    assert support_value is torchrl_ppo.torch.distributions.constraints.real
