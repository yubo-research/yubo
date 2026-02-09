from rl.algos import torchrl_ppo


def test_ppo_tanhnormal_support_property():
    dist = torchrl_ppo._TanhNormal(
        loc=torchrl_ppo.torch.zeros(1),
        scale=torchrl_ppo.torch.ones(1),
    )
    assert dist.support is torchrl_ppo.torch.distributions.constraints.real
