from collections.abc import Callable

import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.lr_scheduler import ConstantLR
from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.submodule_perturbator import SubmodulePerturbator
from optimizer.uhd_mezo import UHDMeZO


class UHDLoop:
    def __init__(
        self,
        module: nn.Module,
        evaluate_fn: Callable[[int], tuple[float, float]],
        *,
        num_iterations: int,
        lr: float = 0.001,
        sigma: float = 0.001,
        weight_decay: float = 0.0,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
        accuracy_fn: Callable[[], float] | None = None,
    ):
        self._module = module
        self._evaluate_fn = evaluate_fn
        self._num_iterations = num_iterations
        self._accuracy_fn = accuracy_fn

        dim = sum(p.numel() for p in module.parameters())
        if num_module_target is not None:
            perturbator = SubmodulePerturbator(module, num_module_target=num_module_target)
        elif num_dim_target is not None:
            perturbator = SparseGaussianPerturbator(module, num_dim_target=num_dim_target)
        else:
            perturbator = GaussianPerturbator(module)
        lr_scheduler = ConstantLR(lr)
        self._uhd = UHDMeZO(perturbator, dim=dim, lr_scheduler=lr_scheduler, sigma=sigma, weight_decay=weight_decay)

    def run(self) -> None:
        num_params = sum(p.numel() for p in self._module.parameters())
        print(f"UHD: num_params = {num_params}")
        last_iter = self._num_iterations - 1
        acc = None
        for i_iter in range(self._num_iterations):
            self._uhd.ask()
            mu, se = self._evaluate_fn(self._uhd.eval_seed)
            self._uhd.tell(mu, se)
            all_params = torch.cat([p.data.reshape(-1) for p in self._module.parameters()])
            mean_param = float(all_params.mean())
            std = float(all_params.std())
            y_best = self._uhd.y_best
            y_best_str = f"{y_best:.4f}" if y_best is not None else "N/A"
            if self._accuracy_fn is not None and (i_iter % 1000 == 0 or i_iter == last_iter or acc is None):
                acc = self._accuracy_fn()
            line = f"EVAL: i_iter = {i_iter} mean_param = {mean_param:.6f} std_param = {std:.4f} sigma = {self._uhd.sigma:.6f} mu = {self._uhd.mu_avg:.4f} se = {self._uhd.se_avg:.4f} y_best = {y_best_str}"
            if acc is not None:
                line += f" test_acc = {acc:.4f}"
            print(line)
