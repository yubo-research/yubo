import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf


def default_bounds(num_dim):
    return torch.tensor([[0.0] * num_dim, [1.0] * num_dim])


def torch_random_choice(x, k=1, replace=True):
    if replace:
        i = torch.randint(len(x), (k,))
    else:
        i = np.random.choice(np.arange(len(x)), k)
    return x[i]


def keep_data(data, keep_style, num_keep):
    if keep_style is None:
        return data

    if keep_style == "trailing":
        if not data:
            return data
        best_idx = max(range(len(data)), key=lambda i: data[i].trajectory.rreturn)
        if best_idx < len(data) - num_keep:
            data_list = list(data[-num_keep + 1 :])
            data_list.append(data[best_idx])
            return data_list
        return data[-num_keep:]
    elif keep_style == "best":
        return sorted(data, key=lambda x: x.trajectory.rreturn, reverse=True)[:num_keep]
    else:
        assert False, keep_style


def keep_trailing(Y, num_keep):
    return np.arange(len(Y))[-num_keep:]


def keep_best(Y, num_keep):
    num_keep_best = num_keep
    return torch.topk(Y, k=num_keep_best).indices.numpy().tolist()


def keep_some(Y, num_keep):
    num_keep_best = num_keep // 2
    num_keep_rest = num_keep - num_keep_best

    i_all = torch.arange(len(Y)).numpy().tolist()
    i_best = torch.topk(Y, k=num_keep_best).indices.numpy().tolist()

    i_rest = list(set(i_all) - set(i_best))
    np.random.shuffle(i_rest)
    i_rest = i_rest[:num_keep_rest]

    idx = list(set(i_best) | set(i_rest))
    idx = sorted(np.unique(idx).tolist())
    assert len(idx) == len(i_best) + len(i_rest)
    assert len(idx) == num_keep, len(idx)

    return idx


def rebound(X, rebounds):
    # X is in [0,1]^d
    lb = rebounds[0, :]
    ub = rebounds[1, :]
    X_r = (X - lb) / (ub - lb)
    return X_r


def unrebound(X_r, rebounds):
    # X_r is in rebounds
    lb = rebounds[0, :]
    ub = rebounds[1, :]
    return lb + (ub - lb) * X_r


def calc_p_max_from_Y(Y):
    is_best = torch.argmax(Y, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)
    p_max = torch.zeros(Y.shape[-1])
    p_max[idcs] = counts / Y.shape[0]
    return p_max


def calc_p_max(model, X, num_Y_samples):
    mvn = model.posterior(X)
    Y = mvn.sample(torch.Size([num_Y_samples])).squeeze()
    assert torch.all((X >= 0) & (X <= 1))
    return calc_p_max_from_Y(Y)


def find_max(model, bounds=None):
    X = model.train_inputs[0]
    Y = model.train_targets
    Y_est = model.posterior(X).mean

    num_dim = X.shape[1]

    if bounds is None:
        bounds = torch.tensor(
            [[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype
        )

    num_ic = 30
    batch_initial_conditions = torch.rand(size=torch.Size((num_ic, 1, num_dim))).to(X)

    k = min(num_ic, len(Y))
    i = torch.topk(Y, k=k).indices
    batch_initial_conditions[:k] = X[i, :].unsqueeze(-2)

    X_max, _ = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        # num_restarts=100,
        # raw_samples=512,
        # options={"batch_limit": 10, "maxiter": 200},
        num_restarts=3,
        options={"batch_limit": num_ic, "maxiter": 100},
        batch_initial_conditions=batch_initial_conditions,
    )

    Y_max = model.posterior(X_max).mean.squeeze()
    if len(model.train_targets) > 0:
        i = torch.argmax(Y_est)
        if Y_est[i] > Y_max:
            X_max = X[i, :][:, None].T

    return X_max.to(X)
