import json
import time

import numpy as np

from third_party.enn.turbo.config.enn_surrogate_config import (
    ENNFitConfig,
    ENNSurrogateConfig,
)
from third_party.enn.turbo.config.factory import turbo_enn_config
from third_party.enn.turbo.optimizer import create_optimizer


def run_scaling_test(num_dim=10, max_n=10000):
    rng = np.random.default_rng(42)
    bounds = np.array([[0.0, 1.0]] * num_dim)

    enn = ENNSurrogateConfig(
        k=10, fit=ENNFitConfig(num_fit_samples=100, num_fit_candidates=100)
    )

    # Use a single optimizer instance for the whole run
    config = turbo_enn_config(enn=enn, num_init=1)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)

    recorded_ns = []
    cum_ask_times = []
    cum_tell_times = []

    total_ask_time = 0.0
    total_tell_time = 0.0

    # Target points: 1, 345, 690, ..., 10000 (30 points)
    target_ns = np.linspace(1, max_n, 30, dtype=int).tolist()
    target_ns = sorted(list(set(target_ns)))
    target_idx = 0

    for n in range(1, max_n + 1):
        # Time ask()
        t0 = time.perf_counter()
        x_next = opt.ask(1)
        t1 = time.perf_counter()
        total_ask_time += t1 - t0

        # Fake data for tell()
        y_obs = rng.standard_normal((1, 1))

        # Time tell()
        t0 = time.perf_counter()
        opt.tell(x_next, y_obs)
        t1 = time.perf_counter()
        total_tell_time += t1 - t0

        # Record at lattice points
        if target_idx < len(target_ns) and n == target_ns[target_idx]:
            recorded_ns.append(n)
            cum_ask_times.append(total_ask_time)
            cum_tell_times.append(total_tell_time)
            with open("enn_scaling_results.json", "w") as f:
                json.dump(
                    {
                        "ns": recorded_ns,
                        "cum_ask_times": cum_ask_times,
                        "cum_tell_times": cum_tell_times,
                    },
                    f,
                )
            print(
                f"N={n}, cum_ask={total_ask_time:.4f}s, cum_tell={total_tell_time:.4f}s",
                flush=True,
            )
            target_idx += 1

    results = {
        "ns": recorded_ns,
        "cum_ask_times": cum_ask_times,
        "cum_tell_times": cum_tell_times,
    }

    with open("enn_scaling_results.json", "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    run_scaling_test()
