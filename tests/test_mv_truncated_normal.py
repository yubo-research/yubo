import torch


def test_mv_truncated_normal_init():
    from sampling.mv_truncated_normal import MVTruncatedNormal

    loc = torch.tensor([0.5, 0.5], dtype=torch.float64)
    scale = torch.tensor([0.1, 0.1], dtype=torch.float64)
    mvtn = MVTruncatedNormal(loc, scale)
    assert mvtn is not None
    assert mvtn.num_dim == 2


def test_mv_truncated_normal_log_prob():
    from sampling.mv_truncated_normal import MVTruncatedNormal

    loc = torch.tensor([0.5, 0.5], dtype=torch.float64)
    scale = torch.tensor([0.1, 0.1], dtype=torch.float64)
    mvtn = MVTruncatedNormal(loc, scale)

    X = torch.tensor([[0.4, 0.5], [0.5, 0.5]], dtype=torch.float64)
    lp = mvtn.log_prob(X)
    assert lp.shape == (2,)
    assert torch.all(torch.isfinite(lp))


def test_mv_truncated_normal_unnormed_prob():
    from sampling.mv_truncated_normal import MVTruncatedNormal

    loc = torch.tensor([0.5, 0.5], dtype=torch.float64)
    scale = torch.tensor([0.1, 0.1], dtype=torch.float64)
    mvtn = MVTruncatedNormal(loc, scale)

    X = torch.tensor([[0.4, 0.5], [0.5, 0.5]], dtype=torch.float64)
    prob = mvtn.unnormed_prob(X)
    assert prob.shape == (2,)
    assert torch.all(prob >= 0)


def test_mv_truncated_normal_rsample():
    from sampling.mv_truncated_normal import MVTruncatedNormal

    loc = torch.tensor([0.5, 0.5], dtype=torch.float64)
    scale = torch.tensor([0.1, 0.1], dtype=torch.float64)
    mvtn = MVTruncatedNormal(loc, scale)

    samples = mvtn.rsample(torch.Size([10]))
    assert samples.shape == (10, 2)
    assert torch.all(samples >= 0)
    assert torch.all(samples <= 1)
