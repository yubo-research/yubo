from __future__ import annotations

import torch
from torch import Tensor

from model.enn_gp_t import EpistemicNearestNeighborsGP
from model.enn_likelihood_t import subsample_loglik


def enn_fit(
    model: EpistemicNearestNeighborsGP,
    k_values: Tensor | None = None,
    var_scale_values: Tensor | None = None,
    num_iterations: int = 2,
    P: int = 10,
) -> dict[str, float]:
    assert isinstance(model, EpistemicNearestNeighborsGP)
    train_X = model.train_inputs[0]
    train_Y = model.train_targets
    if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
        y = train_Y.squeeze(-1)
    else:
        y = train_Y
    train_Yvar = model._train_Yvar
    if train_Yvar.ndim == 2 and train_Yvar.shape[-1] == 1:
        y_var = train_Yvar.squeeze(-1)
    else:
        y_var = train_Yvar
    if y_var.shape[0] == 0:
        y_var = torch.zeros_like(y)

    num_obs = len(model._enn)
    max_k = min(100, max(1, num_obs))
    if k_values is None:
        k_values = torch.logspace(
            0,
            torch.log10(torch.tensor(float(max_k), dtype=torch.float64)),
            steps=30,
            dtype=torch.float64,
            device=train_X.device,
        )

    k_list = [int(v) for v in k_values.tolist() if 3 <= int(v) <= max_k]
    if not k_list:
        k_list = [1]
    if var_scale_values is None:
        var_scale_values = torch.logspace(-4, 3, steps=30, dtype=torch.float64, device=train_X.device)
    var_scale_list = var_scale_values.tolist()
    was_training_model = model.training
    was_training_lik = model.likelihood.training
    model.eval()
    model.likelihood.eval()

    for _ in range(num_iterations):
        median_var_scale = torch.tensor(var_scale_list, dtype=torch.float64, device=train_X.device).median().item()
        with torch.no_grad():
            model.set_var_scale(median_var_scale)
        best_k = None
        best_k_mll = None
        with torch.no_grad():
            for k in k_list:
                model.set_k(k)
                value = subsample_loglik(model, train_X, y, y_var, P=P).item()
                # print("k =", k, "var_scale =", median_var_scale, "value =", value)
                if best_k_mll is None or value > best_k_mll:
                    best_k_mll = value
                    best_k = k
        assert best_k is not None
        best_var_scale = None
        best_var_scale_mll = None
        with torch.no_grad():
            model.set_k(best_k)
            for var_scale in var_scale_list:
                model.set_var_scale(var_scale)
                value = subsample_loglik(model, train_X, y, y_var, P=P).item()
                # print("k =", best_k, "var_scale =", var_scale, "value =", value)
                if best_var_scale_mll is None or value > best_var_scale_mll:
                    best_var_scale_mll = value
                    best_var_scale = var_scale
        assert best_var_scale is not None

        with torch.no_grad():
            model.set_var_scale(best_var_scale)
            model.set_k(best_k)
    if not was_training_model:
        model.eval()
    if was_training_lik:
        model.likelihood.train()

    return {"k": float(best_k), "var_scale": float(best_var_scale)}
