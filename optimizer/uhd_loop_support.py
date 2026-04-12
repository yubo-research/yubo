"""Logging and early-reject helpers for :class:`UHDLoop`."""

from __future__ import annotations

import torch


class UHDLoopSupportMixin:
    def _should_log(self, *, i_iter: int, last_iter: int) -> bool:
        return i_iter == 0 or i_iter == last_iter or self._log_interval <= 1 or (i_iter % self._log_interval == 0)

    def _should_log_enn_stats(self, *, i_iter: int, last_iter: int) -> bool:
        return (self._enn_minus_imputer is not None or self._enn_seed_selector is not None) and self._should_log(i_iter=i_iter, last_iter=last_iter)

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
        if n == 0:
            return None, None
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
                if self._enn_minus_imputer.num_candidates > 1:
                    extra = f" seedselect={self._enn_num_selected}"
                print(
                    f"ENN: real_evals={self._enn_minus_imputer.num_real_evals} imputed_minus={self._enn_minus_imputer.num_imputed} abs_err_ema={err_s}{extra}"
                )
            else:
                print(f"ENN: seedselect={self._enn_num_selected}")
        if self._early_reject_tau is not None:
            print(f"ER: tau={self._early_reject_tau:.4f} skips={self._early_reject_skips}")

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
