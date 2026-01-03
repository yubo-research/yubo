import torch
import torch.nn as nn

from uhd.param_accessor import ModuleParamAccessor, SingleTensorAccessor, make_param_accessor


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 3, bias=False)
        self.b = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.l1(x) + self.b.sum()


def test_single_tensor_accessor_roundtrip():
    t = torch.arange(6.0).view(2, 3).requires_grad_(True)
    acc = SingleTensorAccessor(t)
    assert acc.numel() == 6
    backup = acc.backup()
    delta = torch.ones(6)
    acc.add_inplace_(delta)
    assert torch.allclose(t, backup.view_as(t) + 1.0)
    acc.clamp_(0.0, 5.0)
    assert torch.all(t <= 5.0)
    acc.restore(backup)
    assert torch.allclose(t, backup.view_as(t))


def test_module_param_accessor_mutation_and_restore():
    m = Tiny()
    params = list(m.parameters())
    acc = ModuleParamAccessor(params)
    n = acc.numel()
    assert n == sum(p.numel() for p in params)
    backup = acc.backup()
    delta = torch.randn(n)
    acc.add_inplace_(delta)
    after = acc.clone_flat()
    assert torch.allclose(after, backup + delta)
    acc.restore(backup)
    restored = acc.clone_flat()
    assert torch.allclose(restored, backup)


def test_make_param_accessor_works_for_tensor_and_module():
    t = torch.zeros(5)
    a1 = make_param_accessor(nn.Parameter(t))
    assert a1.numel() == 5
    m = Tiny()
    a2 = make_param_accessor(m)
    assert a2.numel() == sum(p.numel() for p in m.parameters())
