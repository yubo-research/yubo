"""Modal worker + local driver for :func:`benchmark_synthetic_sine_surrogates`.

Each remote invocation runs one ``(N, D, function_name, problem_seed)`` config; the
local entrypoint writes JSON under ``results/`` (or ``--output-dir``).

**CLI (important):** Modal turns ``local_entrypoint`` parameters into **options** (not
bare positional tokens). Pass the benchmark tag with ``--target`` **on the same
``modal run`` line** as the script path—do not insert a lone ``--`` before the options
(that can strip them and yield "Missing option '--target'"). Example::

    modal run experiments/modal_synthetic_sine_benchmark.py \\
      --target sphere --n 12 --d 2 --problem-seed 0 --output-dir results/_xxx

**Batch:** define parameterless functions returning ``list[SyntheticBenchJob]`` in
:mod:`analysis.fitting_time.batch_jobs`, then::

    modal run experiments/modal_synthetic_sine_benchmark.py::batch \\
      --jobs-fn example_sphere_n12_d2_seed0 --output-dir results/_xxx

You can also loop in the shell (separate Modal cold-starts per job)::

    for t in sphere ackley; do
      modal run experiments/modal_synthetic_sine_benchmark.py \\
        --target "$t" --n 12 --d 2 --problem-seed 0 --output-dir results/_xxx
    done

Use ``--target sine`` for the FittingTime-style target, or ``sphere`` / ``ackley``
etc. for :mod:`problems.pure_functions`.

Runs **CPU-only**: no ``gpu=`` request, and ``cpu=2.0`` requests two fractional cores
(Modal API; the benchmark plan referred to this as CPU-only scheduling).

Pure JSON/slug logic lives in :mod:`experiments.synthetic_sine_benchmark_payload` so
imports of that module do not run :func:`experiments.modal_image.mk_image`.

Delegation uses the ``synthetic_sine_benchmark_payload`` **module** (not a stale
import of ``run_synthetic_sine_benchmark_modal_to_disk``) so tests can monkeypatch
payload helpers and have the wrapper pick up the replacement.
"""

from __future__ import annotations

from pathlib import Path

import modal

from experiments import synthetic_sine_benchmark_payload as ssbp
from experiments.modal_image import mk_image

_APP_NAME = "yubo-synthetic-sine-surrogate-benchmark"
_modal_image = mk_image()

app = modal.App(name=_APP_NAME)


@app.function(
    image=_modal_image,
    timeout=60 * 60,
    memory=8192,
    cpu=2.0,
)
def run_synthetic_sine_benchmark_remote(n: int, d: int, function_name: str, problem_seed: int, num_reps: int = 1) -> dict:
    return ssbp.build_synthetic_sine_benchmark_remote_payload(n, d, function_name, problem_seed, num_reps)


def run_synthetic_sine_benchmark_modal_to_disk(
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    output_dir: str | Path,
    *,
    remote_fn=run_synthetic_sine_benchmark_remote,
    num_reps: int = 1,
) -> Path:
    """Fetch one benchmark payload from Modal and write ``results/<slug>.json``."""
    return ssbp.run_synthetic_sine_benchmark_modal_to_disk(
        n,
        d,
        function_name,
        problem_seed,
        output_dir,
        app=app,
        remote_fn=remote_fn,
        start_app=False,
        num_reps=num_reps,
    )


@app.local_entrypoint()
def main(
    target: str,
    n: int = 28,
    d: int = 2,
    problem_seed: int = 0,
    output_dir: str = "results/synthetic_sine_benchmark",
    num_reps: int = 1,
):
    """``target`` is the synthetic benchmark name (same as ``function_name`` in :mod:`evaluate`)."""
    dest = run_synthetic_sine_benchmark_modal_to_disk(n, d, target, problem_seed, output_dir, num_reps=num_reps)
    print(f"wrote {dest.resolve()}")


@app.local_entrypoint()
def batch(
    jobs_fn: str,
    output_dir: str = "results/synthetic_sine_benchmark",
):
    """Run every job from :func:`analysis.fitting_time.batch_jobs` named ``jobs_fn``."""
    jobs = ssbp.load_synthetic_sine_benchmark_jobs(jobs_fn)
    for n, d, fn, problem_seed in jobs:
        dest = run_synthetic_sine_benchmark_modal_to_disk(n, d, fn, problem_seed, output_dir, num_reps=1)
        print(f"wrote {dest.resolve()}")


META_KEY = ssbp.META_KEY
build_synthetic_sine_benchmark_remote_payload = ssbp.build_synthetic_sine_benchmark_remote_payload
load_synthetic_sine_benchmark_jobs = ssbp.load_synthetic_sine_benchmark_jobs
read_synthetic_sine_benchmark_json = ssbp.read_synthetic_sine_benchmark_json
synthetic_sine_benchmark_config_slug = ssbp.synthetic_sine_benchmark_config_slug
synthetic_sine_benchmark_from_payload = ssbp.synthetic_sine_benchmark_from_payload
synthetic_sine_benchmark_result_to_payload = ssbp.synthetic_sine_benchmark_result_to_payload
write_synthetic_sine_benchmark_json = ssbp.write_synthetic_sine_benchmark_json

__all__ = [
    "META_KEY",
    "app",
    "batch",
    "build_synthetic_sine_benchmark_remote_payload",
    "load_synthetic_sine_benchmark_jobs",
    "main",
    "read_synthetic_sine_benchmark_json",
    "run_synthetic_sine_benchmark_modal_to_disk",
    "run_synthetic_sine_benchmark_remote",
    "synthetic_sine_benchmark_config_slug",
    "synthetic_sine_benchmark_from_payload",
    "synthetic_sine_benchmark_result_to_payload",
    "write_synthetic_sine_benchmark_json",
]
