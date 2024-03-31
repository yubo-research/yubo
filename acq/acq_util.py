import torch
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf


def find_max(model, bounds):
    x_cand, _ = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 10, "maxiter": 200},
    )

    Y_cand = model.posterior(x_cand).mean
    if len(model.train_targets) > 0:
        i = torch.argmax(model.train_targets)
        Y_tgt = model.posterior(model.train_inputs[0][i][:, None].T).mean
        if Y_tgt > Y_cand:
            x_cand = model.train_inputs[0][i, :][:, None].T

    return x_cand
