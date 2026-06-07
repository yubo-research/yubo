#!/usr/bin/env python3
"""Batch UHD runs: local multiprocessing or Modal.

Usage:
  ./ops/uhd_batch.py deploy
  ./ops/uhd_batch.py submit --prep module.path.prep_fn [--results-dir DIR]
  ./ops/uhd_batch.py submit --config config.toml --num-reps N [--results-dir DIR]
  ./ops/uhd_batch.py collect [--results-dir DIR]
  ./ops/uhd_batch.py status
  ./ops/uhd_batch.py stop
  ./ops/uhd_batch.py local <config.toml> --num-reps N [--workers W] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imports after repo-root bootstrap (script invoked as ./ops/uhd_batch.py).
from ops.uhd_batch_cli import (  # noqa: E402
    cli,
    collect_cmd,
    deploy_cmd,
    local_cmd,
    status_cmd,
    stop_cmd,
    submit_cmd,
)
from ops.uhd_batch_core import (  # noqa: E402
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
from ops.uhd_batch_local import _batch_local, _local_worker  # noqa: E402
from ops.uhd_batch_modal import (  # noqa: E402
    _HAS_MODAL,
    _UHD_BATCH_DICTS,
    _batch_modal,
    _collect,
    _deploy_uhd_batch_app,
    _ensure_uhd_batch_app,
    _require_modal,
    _results_dict,
    _stop_uhd_batch,
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
    "_UHD_BATCH_DICTS",
    "_batch_modal",
    "_collect",
    "_config_hash",
    "_dict_to_toml",
    "_deploy_uhd_batch_app",
    "_ensure_uhd_batch_app",
    "_experiment_dir",
    "_gen_missing_reps",
    "_load_toml",
    "_parse_eval_lines",
    "_require_modal",
    "_results_dict",
    "_stop_uhd_batch",
    "_submitted_dict",
    "_trace_path",
    "_write_config",
    "_write_trace",
    "app",
    "batch_app",
    "cli",
    "collect_cmd",
    "deploy_cmd",
    "local_cmd",
    "status_cmd",
    "stop_cmd",
    "submit_cmd",
    "uhd_batch_deleter",
    "uhd_batch_resubmitter",
    "uhd_batch_worker",
    "_batch_local",
    "_local_worker",
]

if __name__ == "__main__":
    cli()
