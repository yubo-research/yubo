"""Modal worker + local driver for :func:`benchmark_synthetic_sine_surrogates`.

Each remote invocation runs one ``(N, D, function_name, problem_seed)`` config; the
local entrypoint writes JSON under ``results/`` (or ``--output-dir``).

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
def run_synthetic_sine_benchmark_remote(n: int, d: int, function_name: str | None, problem_seed: int) -> dict:
    return ssbp.build_synthetic_sine_benchmark_remote_payload(n, d, function_name, problem_seed)


def run_synthetic_sine_benchmark_modal_to_disk(
    n: int,
    d: int,
    function_name: str | None,
    problem_seed: int,
    output_dir: str | Path,
    *,
    remote_fn=run_synthetic_sine_benchmark_remote,
) -> Path:
    """Fetch one benchmark payload from Modal and write ``results/<slug>.json``."""
    return ssbp.run_synthetic_sine_benchmark_modal_to_disk(n, d, function_name, problem_seed, output_dir, app=app, remote_fn=remote_fn)


@app.local_entrypoint()
def main(
    n: int = 28,
    d: int = 2,
    function_name: str = "",
    problem_seed: int = 0,
    output_dir: str = "results/synthetic_sine_benchmark",
):
    dest = run_synthetic_sine_benchmark_modal_to_disk(n, d, function_name, problem_seed, output_dir)
    print(f"wrote {dest.resolve()}")


META_KEY = ssbp.META_KEY
build_synthetic_sine_benchmark_remote_payload = ssbp.build_synthetic_sine_benchmark_remote_payload
read_synthetic_sine_benchmark_json = ssbp.read_synthetic_sine_benchmark_json
synthetic_sine_benchmark_config_slug = ssbp.synthetic_sine_benchmark_config_slug
synthetic_sine_benchmark_from_payload = ssbp.synthetic_sine_benchmark_from_payload
synthetic_sine_benchmark_result_to_payload = ssbp.synthetic_sine_benchmark_result_to_payload
write_synthetic_sine_benchmark_json = ssbp.write_synthetic_sine_benchmark_json

__all__ = [
    "META_KEY",
    "app",
    "build_synthetic_sine_benchmark_remote_payload",
    "main",
    "read_synthetic_sine_benchmark_json",
    "run_synthetic_sine_benchmark_modal_to_disk",
    "run_synthetic_sine_benchmark_remote",
    "synthetic_sine_benchmark_config_slug",
    "synthetic_sine_benchmark_from_payload",
    "synthetic_sine_benchmark_result_to_payload",
    "write_synthetic_sine_benchmark_json",
]
