import multiprocessing as mp

import torch

from analysis.data_io import data_is_done, data_writer
from common.seed_all import seed_all
from experiments.experiment_sampler_dispatch import (
    _scan_local_parallel,
    extract_trace_fns,
    post_process,
    post_process_stdout,
    scan_local,
)
from experiments.experiment_sampler_jobs import (
    count_local_trace_jobs,
    mk_replicates,
    prep_args_1,
    prep_d_args,
    sampler,
)
from experiments.experiment_sampler_sampling import sample_1
from experiments.experiment_sampler_types import (
    TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS,
    ExperimentConfig,
    RunConfig,
    _SampleResult,
    true_false,
)
from experiments.experiment_util import ensure_parent


def _register_patch_targets_for_tests() -> None:
    _ = (mp, torch, data_is_done, data_writer, seed_all, ensure_parent)


_register_patch_targets_for_tests()

__all__ = [
    "TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS",
    "ExperimentConfig",
    "RunConfig",
    "_SampleResult",
    "_scan_local_parallel",
    "count_local_trace_jobs",
    "extract_trace_fns",
    "mk_replicates",
    "post_process",
    "post_process_stdout",
    "prep_args_1",
    "prep_d_args",
    "sample_1",
    "sampler",
    "scan_local",
    "true_false",
]
