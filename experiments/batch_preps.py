"""Lazy re-exports of batch prep builders (keeps kiss dependency depth shallow)."""

from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "_RUN_OTHERS_NONFAIL_CELLS": (
        "experiments.batch_preps_timing",
        "_RUN_OTHERS_NONFAIL_CELLS",
    ),
    "prep_ant": ("experiments.batch_preps_rl_sweeps", "prep_ant"),
    "prep_bw": ("experiments.batch_preps_rebuttal", "prep_bw"),
    "prep_cum_time_dim": ("experiments.batch_preps_bench", "prep_cum_time_dim"),
    "prep_cum_time_obs": ("experiments.batch_preps_bench", "prep_cum_time_obs"),
    "prep_dna": ("experiments.batch_preps_rebuttal", "prep_dna"),
    "prep_hop": ("experiments.batch_preps_rebuttal", "prep_hop"),
    "prep_human": ("experiments.batch_preps_rl_sweeps", "prep_human"),
    "prep_leukemia": ("experiments.batch_preps_rebuttal", "prep_leukemia"),
    "prep_mtv_repro": ("experiments.batch_preps_bench", "prep_mtv_repro"),
    "prep_push": ("experiments.batch_preps_seq_sweeps", "prep_push"),
    "prep_rl_one": ("experiments.batch_preps_rl_sweeps", "prep_rl_one"),
    "prep_run_others": ("experiments.batch_preps_timing", "prep_run_others"),
    "prep_seq": ("experiments.batch_preps_seq_sweeps", "prep_seq"),
    "prep_sweep_k": ("experiments.batch_preps_seq_sweeps", "prep_sweep_k"),
    "prep_sweep_k_bw": ("experiments.batch_preps_rl_sweeps", "prep_sweep_k_bw"),
    "prep_sweep_k_tlunar": ("experiments.batch_preps_rl_sweeps", "prep_sweep_k_tlunar"),
    "prep_sweep_p": ("experiments.batch_preps_seq_sweeps", "prep_sweep_p"),
    "prep_sweep_p_bw": ("experiments.batch_preps_rl_sweeps", "prep_sweep_p_bw"),
    "prep_sweep_p_tlunar": ("experiments.batch_preps_rl_sweeps", "prep_sweep_p_tlunar"),
    "prep_sweep_q": ("experiments.batch_preps_bench", "prep_sweep_q"),
    "prep_timing_sweep": ("experiments.batch_preps_timing", "prep_timing_sweep"),
    "prep_tlunar": ("experiments.batch_preps_rebuttal", "prep_tlunar"),
    "prep_validate": ("experiments.batch_preps_rebuttal", "prep_validate"),
    "prep_ts_hd": ("experiments.batch_preps_bench", "prep_ts_hd"),
    "prep_ts_sweep": ("experiments.batch_preps_bench", "prep_ts_sweep"),
    "prep_turbo_abl": ("experiments.batch_preps_timing", "prep_turbo_abl"),
    "prep_turbo_ackley_repro": (
        "experiments.batch_preps_bench",
        "prep_turbo_ackley_repro",
    ),
}
__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod, attr = spec
    _ns: dict[str, Any] = {}
    exec(f"from {mod} import {attr} as _v", _ns)  # noqa: S102
    val = _ns["_v"]
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(__all__)
