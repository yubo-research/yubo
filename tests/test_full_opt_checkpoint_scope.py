from __future__ import annotations

import pytest

from analysis.fitting_time.fitting_time_enn_full_opt import (
    FULL_OPT_MAX_N,
    resolve_full_opt_checkpoints,
)
from analysis.fitting_time.fitting_time_enn_incremental import enn_incremental_checkpoint_ns


def test_resolve_full_opt_checkpoints_default():
    assert resolve_full_opt_checkpoints(None)[-1] == FULL_OPT_MAX_N


def test_full_opt_incremental_checkpoint_csv_accepted_at_cap():
    csv = ",".join(str(n) for n in enn_incremental_checkpoint_ns())
    resolved = resolve_full_opt_checkpoints(csv)
    assert resolved[-1] == FULL_OPT_MAX_N


def test_full_opt_explicit_checkpoint_csv_must_not_exceed_max_n():
    with pytest.raises(ValueError, match=str(FULL_OPT_MAX_N)):
        resolve_full_opt_checkpoints(f"{FULL_OPT_MAX_N + 1}")
