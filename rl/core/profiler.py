"""Profiling utilities for TorchRL training loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import torch


def run_with_profiler(
    config: Any,
    collector: Iterator,
    run_iteration: Callable[[int, Any], None],
    *,
    device: torch.device,
    num_iterations: int,
    start_iteration: int,
) -> None:
    """Run training loop under torch.profiler, export Chrome trace to exp_dir/profile_trace.json."""
    profile_wait = int(getattr(config, "profile_wait", 0))
    profile_warmup = int(getattr(config, "profile_warmup", 1))
    profile_active = int(getattr(config, "profile_active", 3))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif device.type == "mps" and hasattr(torch.profiler.ProfilerActivity, "MPS"):
        activities.append(torch.profiler.ProfilerActivity.MPS)

    schedule = torch.profiler.schedule(
        wait=profile_wait,
        warmup=profile_warmup,
        active=profile_active,
        repeat=0,
    )
    trace_path = Path(config.exp_dir) / "profile_trace.json"
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
        record_shapes=True,
    ) as prof:
        for iteration, batch in enumerate(collector, start=start_iteration + 1):
            if iteration > num_iterations:
                break
            prof.step()
            run_iteration(iteration, batch)

    print(f"[rl/ppo] profile saved to {trace_path}", flush=True)
