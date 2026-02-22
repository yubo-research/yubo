import time
from collections.abc import Callable

import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.lr_scheduler import ConstantLR
from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.submodule_perturbator import SubmodulePerturbator
from optimizer.uhd_enn_imputer import ENNMinusImputer
from optimizer.uhd_enn_seed_selector import ENNMuPlusSeedSelector
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
        enn_minus_imputer: ENNMinusImputer | None = None,
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
        self._enn_minus_imputer = enn_minus_imputer
        self._enn_seed_selector: ENNMuPlusSeedSelector | None = None
        self._enn_num_selected = 0
        self._early_reject_tau: float | None = None
        self._early_reject_mode: str = "y_best"  # "y_best" | "ema"
        self._early_reject_ema_beta: float = 0.99
        self._early_reject_warmup_pos: int = 200
        self._early_reject_mu_plus_ema: float | None = None
        self._early_reject_num_pos_seen: int = 0
        self._early_reject_quantile: float = 0.5
        self._early_reject_window: int = 200
        self._early_reject_mu_plus_window: list[float] = []
        self._early_reject_skips = 0

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

    def _should_log_enn_stats(self, *, i_iter: int, last_iter: int) -> bool:
        return (self._enn_minus_imputer is not None or self._enn_seed_selector is not None) and self._should_log(i_iter=i_iter, last_iter=last_iter)

    @property
    def perturbator(self) -> GaussianPerturbator:
        return self._uhd.perturbator

    def set_enn(
        self,
        *,
        minus_imputer: ENNMinusImputer | None = None,
        seed_selector: ENNMuPlusSeedSelector | None = None,
    ) -> None:
        self._enn_minus_imputer = minus_imputer
        self._enn_seed_selector = seed_selector

    def set_early_reject_advanced(
        self,
        *,
        tau: float | None,
        mode: str = "y_best",
        ema_beta: float = 0.99,
        warmup_pos: int = 200,
        quantile: float = 0.5,
        window: int = 200,
    ) -> None:
        self._early_reject_tau = None if tau is None else float(tau)
        self._early_reject_mode = str(mode)
        self._early_reject_ema_beta = float(ema_beta)
        self._early_reject_warmup_pos = int(warmup_pos)
        self._early_reject_quantile = float(quantile)
        self._early_reject_window = int(window)
        self._early_reject_mu_plus_ema = None
        self._early_reject_num_pos_seen = 0
        self._early_reject_mu_plus_window = []

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
            self._maybe_select_seed()
            self._uhd.ask()
            if self._enn_minus_imputer is not None and self._uhd.positive_phase:
                # The perturbation for this pair is fully defined by (seed, sigma).
                self._enn_minus_imputer.begin_pair(seed=self._uhd.step_seed, sigma=self._uhd.step_sigma)
            phase_is_pos = self._uhd.positive_phase
            mu, se = self._eval_phase(phase_is_pos=phase_is_pos)
            self._uhd.tell(mu, se)
            if phase_is_pos and self._enn_seed_selector is not None:
                self._enn_seed_selector.tell_mu_plus(mu_plus=float(mu))
            early_rejected = False
            if phase_is_pos:
                self._update_early_reject_state(mu_plus=float(mu))
                if self._should_early_reject(mu_plus=float(mu)):
                    self._early_reject_skips += 1
                    self._uhd.skip_negative()
                    early_rejected = True
            if self._enn_minus_imputer is not None and not phase_is_pos:
                self._enn_minus_imputer.update_base_after_step(
                    step_scale=self._uhd.last_step_scale,
                    sigma=self._uhd.step_sigma,
                )
            num_done = i_iter + 1
            if self._should_log(i_iter=i_iter, last_iter=last_iter):
                mean_param, std_param = self._maybe_compute_param_stats()
                y_best = self._uhd.y_best
                y_best_str = f"{y_best:.4f}" if y_best is not None else "N/A"
                acc = self._maybe_update_accuracy(i_iter=i_iter, last_iter=last_iter, acc=acc)
                self._print_log_block(
                    i_iter=i_iter,
                    last_iter=last_iter,
                    y_best_str=y_best_str,
                    acc=acc,
                    mean_param=mean_param,
                    std_param=std_param,
                )

                if self._target_accuracy is not None and acc is not None and acc >= self._target_accuracy:
                    elapsed = time.perf_counter() - t0
                    print(f"UHD: target_accuracy reached: {acc:.4f} >= {self._target_accuracy:.4f} at i_iter={i_iter} (elapsed={elapsed:.2f}s)")
                    break
            if early_rejected:
                continue
        if self._print_summary:
            elapsed = time.perf_counter() - t0
            print(f"UHD: elapsed = {elapsed:.2f}s ({num_done} iterations)")

    def _maybe_select_seed(self) -> None:
        if not self._uhd.positive_phase:
            return
        base_seed = self._uhd.eval_seed
        chosen = base_seed
        if self._enn_seed_selector is not None:
            chosen, _ucb = self._enn_seed_selector.choose_seed_ucb(base_seed=base_seed, sigma=self._uhd.sigma)
        elif self._enn_minus_imputer is not None and int(self._enn_minus_imputer._cfg.num_candidates) > 1:
            chosen, _ucb = self._enn_minus_imputer.choose_seed_ucb(base_seed=base_seed, sigma=self._uhd.sigma)
        else:
            return
        if chosen != base_seed:
            self._enn_num_selected += 1
            self._uhd.set_next_seed(chosen)

    def _print_log_block(
        self,
        *,
        i_iter: int,
        last_iter: int,
        y_best_str: str,
        acc: float | None,
        mean_param: float | None,
        std_param: float | None,
    ) -> None:
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
        if self._should_log_enn_stats(i_iter=i_iter, last_iter=last_iter):
            if self._enn_minus_imputer is not None:
                err = self._enn_minus_imputer.abs_err_ema
                err_s = "N/A" if err is None else f"{err:.4f}"
                extra = ""
                if self._enn_minus_imputer._cfg.num_candidates > 1:
                    extra = f" seedselect={self._enn_num_selected}"
                print(
                    f"ENN: real_evals={self._enn_minus_imputer.num_real_evals} imputed_minus={self._enn_minus_imputer.num_imputed} abs_err_ema={err_s}{extra}"
                )
            else:
                print(f"ENN: seedselect={self._enn_num_selected}")
        if self._early_reject_tau is not None:
            print(f"ER: tau={self._early_reject_tau:.4f} skips={self._early_reject_skips}")

    def _eval_phase(self, *, phase_is_pos: bool) -> tuple[float, float]:
        if self._enn_minus_imputer is not None and not phase_is_pos:
            ok, mu_hat, se_hat = self._enn_minus_imputer.try_impute_current()
            if ok:
                return mu_hat, se_hat
            mu, se = self._evaluate_fn(self._uhd.eval_seed)
            self._enn_minus_imputer.calibrate_minus(mu_minus_real=mu)
            self._enn_minus_imputer.tell_real(mu=mu, phase="minus")
            return mu, se
        mu, se = self._evaluate_fn(self._uhd.eval_seed)
        if self._enn_minus_imputer is not None and phase_is_pos:
            self._enn_minus_imputer.tell_real(mu=mu, phase="plus")
        return mu, se

    def _should_early_reject(self, *, mu_plus: float) -> bool:
        if self._early_reject_tau is None:
            return False
        mode = str(self._early_reject_mode)
        if mode == "y_best":
            y_best = self._uhd.y_best
            if y_best is None:
                return False
            return float(mu_plus) < float(y_best) - float(self._early_reject_tau)
        if mode == "ema":
            if self._early_reject_num_pos_seen < int(self._early_reject_warmup_pos):
                return False
            if self._early_reject_mu_plus_ema is None:
                return False
            return float(mu_plus) < float(self._early_reject_mu_plus_ema) - float(self._early_reject_tau)
        if mode == "quantile":
            if self._early_reject_num_pos_seen < int(self._early_reject_warmup_pos):
                return False
            w = self._early_reject_mu_plus_window
            n = len(w)
            if n <= 0:
                return False
            q = float(self._early_reject_quantile)
            # Deterministic nearest-rank quantile in [0, 1].
            q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
            j = int(round(q * float(n - 1)))
            j = 0 if j < 0 else (n - 1 if j > n - 1 else j)
            thresh = sorted(w)[j]
            return float(mu_plus) < float(thresh) - float(self._early_reject_tau)
        raise ValueError(f"Unknown early_reject_mode: {mode!r}")

    def _update_early_reject_state(self, *, mu_plus: float) -> None:
        self._early_reject_num_pos_seen += 1
        b = float(self._early_reject_ema_beta)
        if self._early_reject_window > 0:
            w = self._early_reject_mu_plus_window
            w.append(float(mu_plus))
            if len(w) > int(self._early_reject_window):
                # Maintain a fixed-size FIFO window.
                del w[0 : len(w) - int(self._early_reject_window)]
        if self._early_reject_mu_plus_ema is None:
            self._early_reject_mu_plus_ema = float(mu_plus)
            return
        self._early_reject_mu_plus_ema = b * float(self._early_reject_mu_plus_ema) + (1.0 - b) * float(mu_plus)
