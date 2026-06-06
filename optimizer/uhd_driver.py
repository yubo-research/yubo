import time
from collections.abc import Callable

from torch import nn

from optimizer.gaussian_perturbator import PerturbatorBase
from optimizer.uhd_enn_imputer import ENNMinusImputer
from optimizer.uhd_enn_seed_selector import ENNMuPlusSeedSelector
from optimizer.uhd_loop_support import UHDLoopSupportMixin


class UHDDriver(UHDLoopSupportMixin):
    """Unified training driver for MeZO-family and ES-family UHD optimizers."""

    def __init__(
        self,
        module: nn.Module,
        uhd,
        perturbator: PerturbatorBase,
        evaluate_fn: Callable[[int], tuple[float, float]],
        *,
        optimizer: str,
        num_iterations: int,
        accuracy_fn: Callable[[], float] | None = None,
        log_interval: int = 1,
        log_param_stats: bool = False,
        accuracy_interval: int = 1000,
        target_accuracy: float | None = None,
        print_summary: bool = False,
        enn_minus_imputer: ENNMinusImputer | None = None,
    ):
        self._module = module
        self._uhd = uhd
        self._perturbator = perturbator
        self._evaluate_fn = evaluate_fn
        self._optimizer = str(optimizer)
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
        self._early_reject_mode: str = "y_best"
        self._early_reject_ema_beta: float = 0.99
        self._early_reject_warmup_pos: int = 200
        self._early_reject_mu_plus_ema: float | None = None
        self._early_reject_num_pos_seen: int = 0
        self._early_reject_quantile: float = 0.5
        self._early_reject_window: int = 200
        self._early_reject_mu_plus_window: list[float] = []
        self._early_reject_skips = 0

    @property
    def perturbator(self) -> PerturbatorBase:
        return self._perturbator

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

    def _is_two_phase(self) -> bool:
        return hasattr(self._uhd, "positive_phase")

    def _can_early_reject(self) -> bool:
        return hasattr(self._uhd, "skip_negative")

    def _positive_phase(self) -> bool:
        if self._is_two_phase():
            return bool(self._uhd.positive_phase)
        return True

    def run(self) -> None:
        num_params = sum(p.numel() for p in self._module.parameters())
        print(f"UHD: num_params = {num_params}, optimizer = {self._optimizer}")
        last_iter = self._num_iterations - 1
        acc = None
        t0 = time.perf_counter()
        num_done = 0
        for i_iter in range(self._num_iterations):
            t_prop0 = time.perf_counter()
            self._maybe_select_seed()
            self._uhd.ask()
            proposal_dt = time.perf_counter() - t_prop0
            if self._enn_minus_imputer is not None and self._positive_phase():
                step_seed = getattr(self._uhd, "step_seed", self._uhd.eval_seed)
                step_sigma = getattr(self._uhd, "step_sigma", self._uhd.sigma)
                self._enn_minus_imputer.begin_pair(seed=step_seed, sigma=step_sigma)
            phase_is_pos = self._positive_phase()
            t_eval0 = time.perf_counter()
            mu, se = self._eval_phase(phase_is_pos=phase_is_pos)
            eval_dt = time.perf_counter() - t_eval0
            self._uhd.tell(mu, se)
            if phase_is_pos and self._enn_seed_selector is not None:
                self._enn_seed_selector.tell_mu_plus(mu_plus=float(mu))
            early_rejected = False
            if phase_is_pos and self._can_early_reject():
                self._update_early_reject_state(mu_plus=float(mu))
                if self._should_early_reject(mu_plus=float(mu)):
                    self._early_reject_skips += 1
                    self._uhd.skip_negative()
                    early_rejected = True
            if self._enn_minus_imputer is not None and not phase_is_pos:
                step_sigma = getattr(self._uhd, "step_sigma", self._uhd.sigma)
                step_scale = getattr(self._uhd, "last_step_scale", 0.0)
                self._enn_minus_imputer.update_base_after_step(step_scale=step_scale, sigma=step_sigma)
            num_done = i_iter + 1
            if self._should_log(i_iter=i_iter, last_iter=last_iter):
                mean_param, std_param = self._maybe_compute_param_stats()
                y_best = self._uhd.y_best
                y_best_str = f"{y_best:.4f}" if y_best is not None else "N/A"
                acc = self._maybe_update_accuracy(i_iter=i_iter, last_iter=last_iter, acc=acc)
                self._print_log_block(
                    i_iter=i_iter,
                    last_iter=last_iter,
                    proposal_dt=proposal_dt,
                    eval_dt=eval_dt,
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
        if not self._positive_phase():
            return
        base_seed = self._uhd.eval_seed
        chosen = base_seed
        if self._enn_seed_selector is not None:
            chosen, _ucb = self._enn_seed_selector.choose_seed_ucb(base_seed=base_seed, sigma=self._uhd.sigma)
        elif self._enn_minus_imputer is not None and self._enn_minus_imputer.num_candidates > 1:
            chosen, _ucb = self._enn_minus_imputer.choose_seed_ucb(base_seed=base_seed, sigma=self._uhd.sigma)
        else:
            return
        if chosen != base_seed and hasattr(self._uhd, "set_next_seed"):
            self._enn_num_selected += 1
            self._uhd.set_next_seed(chosen)

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
