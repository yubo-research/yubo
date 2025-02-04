def test_keep_some():
    import torch

    from acq.acq_util import keep_some

    Y = torch.randn(100)

    idx = keep_some(Y, num_keep=30)

    assert len(idx) == 30


def test_find_max():
    import time

    import numpy as np
    import torch

    from acq.acq_util import find_max

    from .test_util import gp_parabola

    data = []

    for i_seed in range(10):
        np.random.seed(170 + i_seed)
        torch.manual_seed(17 + i_seed)
        model, X_0 = gp_parabola(num_samples=10, num_dim=2)
        t_0 = time.time()
        X = find_max(model)
        t_f = time.time()
        d = np.sqrt(((X - X_0).numpy() ** 2).sum())
        data.append((t_f - t_0, d))

    data = np.array(data)

    print()
    print(data)
    print(f"TIME: {data[:,0].mean():.3f}")
    print(f"{data[:,1].mean():.3f} {data[:,1].std():.3f}")


def test_rebound():
    import torch

    from acq.acq_util import rebound

    torch.manual_seed(17)
    X = torch.rand(size=(10, 3))
    rebound(X, torch.tensor([[0.3, 0.4, 0.5], [0.9, 0.8, 0.7]]))
