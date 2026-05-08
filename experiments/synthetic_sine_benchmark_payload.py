"""JSON/slug helpers and remote payload builder for surrogate timing benchmarks.

Each run is tagged by a **required** ``function_name`` (e.g. ``sine`` for the FittingTime
target, or ``sphere`` / ``ackley`` for :mod:`problems.pure_functions`). No Modal image
build: safe to import in tests without ``mk_image()`` cost.
"""

from __future__ import annotations

import modal

from analysis.fitting_time.evaluate import benchmark_synthetic_sine_surrogates
from experiments.synthetic_sine_benchmark_payload_core import (
    META_KEY,
    read_synthetic_sine_benchmark_json,
    synthetic_sine_benchmark_config_slug,
    synthetic_sine_benchmark_from_payload,
    synthetic_sine_benchmark_rep_slug,
    synthetic_sine_benchmark_result_to_payload,
    synthetic_surrogate_benchmark_to_wide_row,
    write_synthetic_sine_benchmark_json,
)
from experiments.synthetic_sine_benchmark_payload_jobs import (
    load_synthetic_sine_benchmark_jobs,
)
from experiments.synthetic_sine_benchmark_payload_remote import (
    build_synthetic_sine_benchmark_remote_payload,
    run_synthetic_sine_benchmark_modal_to_disk,
)
from experiments.synthetic_sine_benchmark_payload_tables import (
    load_synthetic_sine_benchmark_json_dir,
    load_synthetic_sine_benchmark_json_dir_long,
    synthetic_surrogate_benchmark_row_caption,
    wide_surrogate_benchmark_row_to_comparison_records,
    wide_surrogate_benchmark_row_to_long_records,
)


def _register_patch_targets_for_tests() -> None:
    _ = (modal, benchmark_synthetic_sine_surrogates)


_register_patch_targets_for_tests()

__all__ = [
    "META_KEY",
    "build_synthetic_sine_benchmark_remote_payload",
    "load_synthetic_sine_benchmark_jobs",
    "load_synthetic_sine_benchmark_json_dir",
    "load_synthetic_sine_benchmark_json_dir_long",
    "read_synthetic_sine_benchmark_json",
    "run_synthetic_sine_benchmark_modal_to_disk",
    "synthetic_surrogate_benchmark_row_caption",
    "synthetic_surrogate_benchmark_to_wide_row",
    "synthetic_sine_benchmark_config_slug",
    "synthetic_sine_benchmark_rep_slug",
    "synthetic_sine_benchmark_from_payload",
    "synthetic_sine_benchmark_result_to_payload",
    "wide_surrogate_benchmark_row_to_long_records",
    "wide_surrogate_benchmark_row_to_comparison_records",
    "write_synthetic_sine_benchmark_json",
]
