import torch


def test_y_transform_basic():
    from acq.y_transform import YTransform

    transform = YTransform()
    Y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    Y_transformed = transform(Y)
    assert Y_transformed.shape == Y.shape
    assert torch.all(torch.isfinite(Y_transformed))


def test_y_transform_single_element():
    from acq.y_transform import YTransform

    transform = YTransform()
    Y = torch.tensor([5.0])
    Y_transformed = transform(Y)
    assert Y_transformed.shape == Y.shape


def test_y_transform_batch_shape():
    from acq.y_transform import YTransform

    transform = YTransform(batch_shape=torch.Size([2]))
    assert transform.batch_shape == torch.Size([2])
