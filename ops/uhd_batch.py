#!/usr/bin/env python3
"""Batch UHD runs: local multiprocessing or Modal.

Usage:
  ./ops/uhd_batch.py local  <config.toml> --num-reps N [--workers W] [--results-dir DIR]
  ./ops/uhd_batch.py modal  <config.toml> --num-reps N [--results-dir DIR]
  ./ops/uhd_batch.py collect [--results-dir DIR]
  ./ops/uhd_batch.py status
  ./ops/uhd_batch.py cleanup
"""

from ops.uhd_batch_cli import (
    batch_cmd,
    cleanup_cmd,
    cli,
    collect_cmd,
    local_cmd,
    modal_cmd,
    status_cmd,
)
from ops.uhd_batch_core import (
    _APP_NAME,
    _DEFAULT_RESULTS,
    _config_hash,
    _dict_to_toml,
    _experiment_dir,
    _gen_missing_reps,
    _load_toml,
    _parse_eval_lines,
    _trace_path,
    _write_config,
    _write_trace,
)
from ops.uhd_batch_local import _batch_local, _local_worker
from ops.uhd_batch_modal import (
    _HAS_MODAL,
    _batch_modal,
    _collect,
    _require_modal,
    _results_dict,
    _submitted_dict,
    app,
    batch_app,
    uhd_batch_deleter,
    uhd_batch_resubmitter,
    uhd_batch_worker,
)

__all__ = [
    "_APP_NAME",
    "_DEFAULT_RESULTS",
    "_HAS_MODAL",
    "_batch_modal",
    "_collect",
    "_config_hash",
    "_dict_to_toml",
    "_experiment_dir",
    "_gen_missing_reps",
    "_load_toml",
    "_parse_eval_lines",
    "_require_modal",
    "_results_dict",
    "_submitted_dict",
    "_trace_path",
    "_write_config",
    "_write_trace",
    "app",
    "batch_app",
    "batch_cmd",
    "cleanup_cmd",
    "cli",
    "collect_cmd",
    "local_cmd",
    "modal_cmd",
    "status_cmd",
    "uhd_batch_deleter",
    "uhd_batch_resubmitter",
    "uhd_batch_worker",
    "_batch_local",
    "_local_worker",
]

if __name__ == "__main__":
    cli()
