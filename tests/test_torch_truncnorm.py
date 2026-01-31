import torch


def test_truncated_standard_normal_init():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    assert dist is not None


def test_truncated_standard_normal_sample():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    samples = dist.rsample(torch.Size([10]))
    assert samples.shape == (10,)
    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 1.0)


def test_truncated_standard_normal_log_prob():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    x = torch.tensor(0.0)
    log_prob = dist.log_prob(x)
    assert torch.isfinite(log_prob)


def test_truncated_standard_normal_mean():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    mean = dist.mean
    assert torch.isfinite(mean)


def test_truncated_standard_normal_support():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    support = dist.support
    assert support is not None


def test_truncated_standard_normal_variance():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    var = dist.variance
    assert torch.isfinite(var)


def test_truncated_standard_normal_entropy():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    entropy = dist.entropy()
    assert torch.isfinite(entropy)


def test_truncated_standard_normal_auc():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    auc = dist.auc
    assert torch.isfinite(auc)


def test_truncated_standard_normal_cdf():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    x = torch.tensor(0.0)
    cdf = dist.cdf(x)
    assert torch.isfinite(cdf)


def test_truncated_standard_normal_icdf():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    p = torch.tensor(0.5)
    icdf = dist.icdf(p)
    assert torch.isfinite(icdf)


def test_truncated_standard_normal_p_and_sample():
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    a = torch.tensor(-1.0)
    b = torch.tensor(1.0)
    dist = TruncatedStandardNormal(a, b)
    result = dist.p_and_sample(torch.Size([10]))
    assert result.X.shape == (10,)
    assert result.pi.shape == (10,)
