from __future__ import annotations

from typing import Any, Callable

from .uhd_be_enn import be_enn_selection_ready


def run_mezo_be_ask(
    self: Any,
    *,
    embed_unselected: Callable[[], Any],
) -> None:
    m = self._mezo
    if m.positive_phase:
        if be_enn_selection_ready(
            obs_count=len(self._zs),
            warmup=self._warmup,
            enn_k=self._enn_k,
            has_params=self._enn_params is not None,
        ):
            best_seed, z_plus, z_minus = self._select_seed()
            m.set_next_seed(best_seed)
            self._z_plus = z_plus
            self._z_minus = z_minus
            self._selected = True
        else:
            self._selected = False
        m.ask()
        if not self._selected:
            self._z_plus = embed_unselected()
    else:
        m.ask()
        if not self._selected:
            self._z_minus = embed_unselected()
