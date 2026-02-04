from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from optimizer.trajectories import collect_trajectory
from problems.env_conf import default_policy, get_env_conf


def _parse_eps(text: str) -> list[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    values = [float(p) for p in parts]
    if not values:
        raise ValueError("eps list must be non-empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"eps values must be > 0, got {values}")
    return values


def _summarize(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _load_env_policy(env_tag: str, seed: int):
    env_conf = get_env_conf(env_tag, problem_seed=seed, noise_seed_0=10 * seed)
    policy = default_policy(env_conf)
    return env_conf, policy


def _collect_states(env_conf, policy, *, num_states: int, seed: int) -> np.ndarray:
    traj = collect_trajectory(env_conf, policy, noise_seed=seed)
    states = np.asarray(traj.states).T
    if states.shape[0] <= num_states:
        return states
    rng = np.random.default_rng(seed)
    idx = rng.choice(states.shape[0], size=num_states, replace=False)
    return states[idx]


def _policy_actions(policy, states: np.ndarray) -> np.ndarray:
    return np.stack([policy(s) for s in states], axis=0)


def _directional_sensitivity(policy, theta_base: np.ndarray, states: np.ndarray, eps_list: list[float], num_dirs: int, seed: int):
    rng = np.random.default_rng(seed)
    base_actions = _policy_actions(policy, states)
    results = {}
    for eps in eps_list:
        deltas = []
        scaled = []
        for _ in range(num_dirs):
            direction = rng.normal(size=theta_base.shape)
            direction /= np.linalg.norm(direction)
            policy.set_params(theta_base + eps * direction)
            actions = _policy_actions(policy, states)
            delta = np.linalg.norm(actions - base_actions, axis=1).mean()
            deltas.append(float(delta))
            scaled.append(float(delta / eps))
        results[str(eps)] = {
            "delta": _summarize(deltas),
            "delta_over_eps": _summarize(scaled),
        }
    policy.set_params(theta_base)
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose theta-to-action sensitivity for a yubo policy.")
    parser.add_argument("--env-tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-states", type=int, default=512)
    parser.add_argument("--num-dirs", type=int, default=50)
    parser.add_argument("--eps", type=str, default="1e-3,1e-2,1e-1")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    env_conf, policy = _load_env_policy(args.env_tag, args.seed)
    states = _collect_states(env_conf, policy, num_states=args.num_states, seed=args.seed)
    theta_base = np.asarray(policy.get_params(), dtype=np.float64)
    eps_list = _parse_eps(args.eps)
    results = _directional_sensitivity(policy, theta_base, states, eps_list, num_dirs=args.num_dirs, seed=args.seed)

    payload = {
        "env_tag": args.env_tag,
        "seed": int(args.seed),
        "num_states": int(states.shape[0]),
        "num_dirs": int(args.num_dirs),
        "eps_list": eps_list,
        "theta_dim": int(theta_base.size),
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
