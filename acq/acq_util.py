import torch
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf


def find_max(model, bounds=None):
    X = model.train_inputs[0]
    Y = model.train_targets.squeeze()
    Y_est = model.posterior(X).mean.squeeze()

    num_dim = X.shape[1]

    if bounds is None:
        bounds = torch.tensor([[0.0] * num_dim, [1.0] * num_dim], device=X.device, dtype=X.dtype)

    num_ic = 30
    batch_initial_conditions = torch.rand(size=torch.Size((num_ic, 1, num_dim)))

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
