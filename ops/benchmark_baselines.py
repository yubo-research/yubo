#!/usr/bin/env python
"""Compute random-policy baselines for BO benchmark envs.

Run: python ops/benchmark_baselines.py [env_tag ...]

If no env_tag given, runs default set: dm:cheetah-run:gauss, dm:hopper-hop:gauss, cheetah-16x16-gauss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add repo root (must run before importing project modules)
_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from optimizer.trajectories import collect_trajectory  # noqa: E402
from problems.env_conf import default_policy, get_env_conf  # noqa: E402


def _main():
    parser = argparse.ArgumentParser(description="Random policy baselines for BO benchmarks")
    parser.add_argument(
        "env_tags",
        nargs="*",
        default=["dm:cheetah-run:gauss", "dm:hopper-hop:gauss", "cheetah-16x16-gauss"],
        help="Env tags (default: dm:cheetah-run:gauss dm:hopper-hop:gauss cheetah-16x16-gauss)",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=20, help="Random policies to evaluate")
    parser.add_argument("--seed", type=int, default=17, help="Problem seed")
    args = parser.parse_args()

    for env_tag in args.env_tags:
        try:
            env_conf = get_env_conf(env_tag, problem_seed=args.seed)
            policy = default_policy(env_conf)
        except Exception as e:
            print(f"{env_tag}: ERROR {e}", flush=True)
            continue

        rets = []
        rng = np.random.default_rng(args.seed)
        for _ in range(args.num_samples):
            x = rng.uniform(-1, 1, size=policy.num_params())
            policy.set_params(x)
            r = collect_trajectory(env_conf, policy).rreturn
            rets.append(float(r))

        rets = np.array(rets)
        mean, std = float(rets.mean()), float(rets.std())
        se = std / np.sqrt(len(rets))
        print(
            f"{env_tag}: random mean={mean:.1f} ± {se:.1f} (std={std:.1f}) min={rets.min():.1f} max={rets.max():.1f} n_params={policy.num_params()}",
            flush=True,
        )


if __name__ == "__main__":
    _main()
