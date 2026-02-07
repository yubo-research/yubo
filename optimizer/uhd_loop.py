from collections.abc import Callable

import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd import UHD


class UHDLoop:
    def __init__(
        self,
        module: nn.Module,
        evaluate_fn: Callable[[], float],
        *,
        sigma_0: float,
        num_iterations: int,
    ):
        self._module = module
        self._evaluate_fn = evaluate_fn
        self._num_iterations = num_iterations

        dim = sum(p.numel() for p in module.parameters())
        perturbator = GaussianPerturbator(module)
        self._uhd = UHD(perturbator, sigma_0=sigma_0, dim=dim)

    def run(self) -> None:
        num_params = sum(p.numel() for p in self._module.parameters())
        print(f"UHD: num_params = {num_params}")
        for i_iter in range(self._num_iterations):
            seed = self._uhd.ask()
            y = self._evaluate_fn()
            self._uhd.tell(seed, y)
            all_params = torch.cat([p.data.reshape(-1) for p in self._module.parameters()])
            max_abs = float(all_params.abs().max())
            std = float(all_params.std())
            print(f"EVAL: i_iter = {i_iter} max_abs_param = {max_abs:.4f} std_param = {std:.4f} sigma = {self._uhd.sigma:.6f} y_max = {self._uhd.y_max}")
