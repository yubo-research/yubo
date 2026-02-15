import time
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
        log_interval: int = 1,
        log_param_stats: bool = False,
        accuracy_interval: int = 1000,
        target_accuracy: float | None = None,
        print_summary: bool = False,
    ):
        self._module = module
        self._evaluate_fn = evaluate_fn
        self._num_iterations = num_iterations
        self._accuracy_fn = accuracy_fn
        self._log_interval = int(log_interval)
        self._log_param_stats = bool(log_param_stats)
        self._accuracy_interval = int(accuracy_interval)
        self._target_accuracy = float(target_accuracy) if target_accuracy is not None else None
        self._print_summary = bool(print_summary)

        dim = sum(p.numel() for p in module.parameters())
        if num_module_target is not None:
            perturbator = SubmodulePerturbator(module, num_module_target=num_module_target)
        elif num_dim_target is not None:
            perturbator = SparseGaussianPerturbator(module, num_dim_target=num_dim_target)
        else:
            perturbator = GaussianPerturbator(module)
        lr_scheduler = ConstantLR(lr)
        self._uhd = UHDMeZO(
            perturbator,
            dim=dim,
            lr_scheduler=lr_scheduler,
            sigma=sigma,
            weight_decay=weight_decay,
        )

    def _should_log(self, *, i_iter: int, last_iter: int) -> bool:
        return i_iter == 0 or i_iter == last_iter or self._log_interval <= 1 or (i_iter % self._log_interval == 0)

    def _maybe_compute_param_stats(self) -> tuple[float | None, float | None]:
        if not self._log_param_stats:
            return None, None

        # WARNING: reductions can cause device sync; keep off by default.
        with torch.no_grad():
            s = 0.0
            ss = 0.0
            n = 0
            for p in self._module.parameters():
                x = p.data.reshape(-1).float()
                s += float(x.sum())
                ss += float((x * x).sum())
                n += x.numel()
        mean_param = s / n
        var = max(ss / n - mean_param * mean_param, 0.0)
        std_param = var**0.5
        return mean_param, std_param

    def _maybe_update_accuracy(self, *, i_iter: int, last_iter: int, acc: float | None) -> float | None:
        if self._accuracy_fn is None:
            return acc
        if i_iter == last_iter or acc is None or self._accuracy_interval <= 1 or (i_iter % self._accuracy_interval == 0):
            return self._accuracy_fn()
        return acc

    def _format_eval_line(
        self,
        *,
        i_iter: int,
        y_best_str: str,
        mu: float,
        se: float,
        acc: float | None,
        mean_param: float | None,
        std_param: float | None,
    ) -> str:
        line = f"EVAL: i_iter = {i_iter} sigma = {self._uhd.sigma:.6f} mu = {mu:.4f} se = {se:.4f} y_best = {y_best_str}"
        if mean_param is not None and std_param is not None:
            line += f" mean_param = {mean_param:.6f} std_param = {std_param:.4f}"
        if acc is not None:
            line += f" test_acc = {acc:.4f}"
        return line

    def run(self) -> None:
        num_params = sum(p.numel() for p in self._module.parameters())
        print(f"UHD: num_params = {num_params}")
        last_iter = self._num_iterations - 1
        acc = None
        t0 = time.perf_counter()
        num_done = 0
        for i_iter in range(self._num_iterations):
            self._uhd.ask()
            mu, se = self._evaluate_fn(self._uhd.eval_seed)
            self._uhd.tell(mu, se)
            num_done = i_iter + 1
            if self._should_log(i_iter=i_iter, last_iter=last_iter):
                mean_param, std_param = self._maybe_compute_param_stats()
                y_best = self._uhd.y_best
                y_best_str = f"{y_best:.4f}" if y_best is not None else "N/A"
                acc = self._maybe_update_accuracy(i_iter=i_iter, last_iter=last_iter, acc=acc)
                print(
                    self._format_eval_line(
                        i_iter=i_iter,
                        y_best_str=y_best_str,
                        mu=self._uhd.mu_avg,
                        se=self._uhd.se_avg,
                        acc=acc,
                        mean_param=mean_param,
                        std_param=std_param,
                    )
                )

                if self._target_accuracy is not None and acc is not None and acc >= self._target_accuracy:
                    elapsed = time.perf_counter() - t0
                    print(f"UHD: target_accuracy reached: {acc:.4f} >= {self._target_accuracy:.4f} at i_iter={i_iter} (elapsed={elapsed:.2f}s)")
                    break
        if self._print_summary:
            elapsed = time.perf_counter() - t0
            print(f"UHD: elapsed = {elapsed:.2f}s ({num_done} iterations)")
