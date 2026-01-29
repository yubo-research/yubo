import torch
import torch.nn as nn


def test_single_tensor_accessor_init():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.randn(5)
    acc = SingleTensorAccessor(t)
    assert acc.numel() == 5
    assert acc.num_dims == 5


def test_single_tensor_accessor_clone_flat():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.tensor([1.0, 2.0, 3.0])
    acc = SingleTensorAccessor(t)
    cloned = acc.clone_flat()
    assert torch.allclose(cloned, t)


def test_single_tensor_accessor_add_():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.tensor([1.0, 2.0, 3.0])
    acc = SingleTensorAccessor(t)
    delta = torch.tensor([0.1, 0.2, 0.3])
    acc.add_(delta)
    assert torch.allclose(t, torch.tensor([1.1, 2.2, 3.3]))


def test_single_tensor_accessor_gather():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    acc = SingleTensorAccessor(t)
    indices = torch.tensor([0, 2, 4])
    gathered = acc.gather(indices)
    assert torch.allclose(gathered, torch.tensor([1.0, 3.0, 5.0]))


def test_single_tensor_accessor_scatter_():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    acc = SingleTensorAccessor(t)
    indices = torch.tensor([0, 2])
    values = torch.tensor([10.0, 30.0])
    acc.scatter_(indices, values)
    assert torch.allclose(t, torch.tensor([10.0, 2.0, 30.0, 4.0, 5.0]))


def test_single_tensor_accessor_mul_():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.tensor([1.0, 2.0, 3.0])
    acc = SingleTensorAccessor(t)
    acc.mul_(2.0)
    assert torch.allclose(t, torch.tensor([2.0, 4.0, 6.0]))


def test_single_tensor_accessor_copy_():
    from uhd.param_accessor import SingleTensorAccessor

    t = torch.tensor([1.0, 2.0, 3.0])
    acc = SingleTensorAccessor(t)
    src = torch.tensor([10.0, 20.0, 30.0])
    acc.copy_(src)
    assert torch.allclose(t, src)


def test_module_param_accessor_init():
    from uhd.param_accessor import ModuleParamAccessor

    model = nn.Linear(3, 2)
    acc = ModuleParamAccessor(model.parameters())
    assert acc.numel() > 0


def test_module_param_accessor_clone_flat():
    from uhd.param_accessor import ModuleParamAccessor

    model = nn.Linear(3, 2)
    acc = ModuleParamAccessor(model.parameters())
    cloned = acc.clone_flat()
    assert cloned.numel() == acc.numel()


def test_make_param_accessor_tensor():
    from uhd.param_accessor import make_param_accessor

    t = torch.randn(5)
    acc = make_param_accessor(t)
    assert acc.numel() == 5


def test_make_param_accessor_module():
    from uhd.param_accessor import make_param_accessor

    model = nn.Linear(3, 2)
    acc = make_param_accessor(model)
    assert acc.numel() > 0
