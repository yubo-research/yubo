from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import modal
except ModuleNotFoundError:

    def _modal_unavailable():
        raise ModuleNotFoundError("modal")

    modal = SimpleNamespace(enable_output=_modal_unavailable)

from analysis.fitting_time.evaluate import normalize_benchmark_function_name
from experiments.synthetic_sine_benchmark_payload_core import (
    synthetic_sine_benchmark_config_slug,
    write_synthetic_sine_benchmark_json,
)


def build_synthetic_sine_benchmark_remote_payload(n: int, d: int, function_name: str, problem_seed: int, num_reps: int = 1) -> dict:
    """Run the benchmark and return JSON-serializable payload (used by the Modal worker)."""
    pl = sys.modules["experiments.synthetic_sine_benchmark_payload"]
    fn = normalize_benchmark_function_name(function_name)
    r = pl.benchmark_synthetic_sine_surrogates(N=n, D=d, function_name=fn, problem_seed=problem_seed, num_reps=int(num_reps))
    return pl.synthetic_sine_benchmark_result_to_payload(r, n=n, d=d, function_name=fn, problem_seed=problem_seed, num_reps=int(num_reps))


def run_synthetic_sine_benchmark_modal_to_disk(
    n: int,
    d: int,
    function_name: str,
    problem_seed: int,
    output_dir: str | Path,
    *,
    app: Any,
    remote_fn: Any,
    start_app: bool = True,
    num_reps: int = 1,
) -> Path:
    """When ``start_app`` is false, skip ``app.run()`` (Modal ``local_entrypoint`` already runs the app)."""
    fn = normalize_benchmark_function_name(function_name)
    out_root = Path(output_dir)
    slug = synthetic_sine_benchmark_config_slug(n=n, d=d, function_name=fn, problem_seed=problem_seed, num_reps=int(num_reps))
    dest = out_root / f"{slug}.json"
    nr = int(num_reps)
    with modal.enable_output():
        if start_app:
            with app.run():
                payload = remote_fn.remote(n, d, fn, int(problem_seed), nr)
        else:
            payload = remote_fn.remote(n, d, fn, int(problem_seed), nr)
    write_synthetic_sine_benchmark_json(dest, payload)
    return dest
