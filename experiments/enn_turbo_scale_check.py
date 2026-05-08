"""Structural ENN TuRBO scale check for very large ambient dimensions.

This intentionally avoids materializing full RAASP candidate matrices.  The
FAST RAASP implementation first samples K=max(1, Binomial(D, 20/D)) changed
coordinates, so this diagnostic samples K directly and uses the ENN trust-region
state for the failure-clock check.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from enn.turbo.config.turbo_tr_config import TurboTRConfig
from enn.turbo.turbo_trust_region import TurboTrustRegion


DEFAULT_DIMS = (10_000, 500_000, 10_000_000, 1_000_000_000)
DEFAULT_QS = (1, 16, 256)


@dataclass(frozen=True)
class ScaleRow:
    num_dim: int
    num_arms: int
    failure_tolerance: int
    formula_tolerance: int
    failure_evals: int
    raasp_mean_mc: float
    raasp_mean_theory: float
    ratio: float
    shrink_halved: bool
    coord_check: bool


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip().replace("_", "")) for part in value.split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("all values must be positive")
    return out


def raasp_mean_theory(num_dim: int, num_pert: int = 20) -> float:
    p = min(float(num_pert) / float(num_dim), 1.0)
    return float(num_dim) * p + math.exp(float(num_dim) * math.log1p(-p))


def sample_raasp_support(
    rng: np.random.Generator,
    *,
    num_dim: int,
    num_samples: int,
    num_pert: int = 20,
) -> np.ndarray:
    p = min(float(num_pert) / float(num_dim), 1.0)
    return np.maximum(rng.binomial(num_dim, p, size=num_samples), 1)


def check_coordinate_sampling(
    rng: np.random.Generator,
    *,
    num_dim: int,
    support_size: int,
) -> bool:
    idx = rng.choice(num_dim, size=int(support_size), replace=False)
    return bool(idx.size == np.unique(idx).size and idx.min(initial=0) >= 0 and idx.max(initial=0) < num_dim)


def check_failure_clock(num_dim: int, num_arms: int) -> tuple[int, int, bool]:
    tr = TurboTrustRegion(TurboTRConfig(), num_dim)
    tr.validate_request(num_arms)
    formula = int(math.ceil(max(4.0 / float(num_arms), float(num_dim) / float(num_arms))))
    if tr.failure_tolerance != formula:
        raise AssertionError((num_dim, num_arms, tr.failure_tolerance, formula))

    tr.best_value = 0.0
    tr.prev_num_obs = num_arms
    tr.failure_counter = tr.failure_tolerance - 1
    before = tr.length
    tr.update(np.zeros(2 * num_arms), np.array([0.0]))
    return tr.failure_tolerance, formula, bool(tr.length == 0.5 * before)


def build_rows(
    *,
    dims: tuple[int, ...],
    qs: tuple[int, ...],
    num_samples: int,
    seed: int,
) -> list[ScaleRow]:
    rng = np.random.default_rng(seed)
    rows: list[ScaleRow] = []
    for num_dim in dims:
        supports = sample_raasp_support(rng, num_dim=num_dim, num_samples=num_samples)
        support_mean_mc = float(np.mean(supports))
        support_mean_theory = raasp_mean_theory(num_dim)
        coord_check = check_coordinate_sampling(
            rng,
            num_dim=num_dim,
            support_size=int(supports[0]),
        )
        for num_arms in qs:
            failure_tolerance, formula_tolerance, shrink_halved = check_failure_clock(
                num_dim,
                num_arms,
            )
            failure_evals = int(num_arms) * int(failure_tolerance)
            rows.append(
                ScaleRow(
                    num_dim=num_dim,
                    num_arms=num_arms,
                    failure_tolerance=failure_tolerance,
                    formula_tolerance=formula_tolerance,
                    failure_evals=failure_evals,
                    raasp_mean_mc=support_mean_mc,
                    raasp_mean_theory=support_mean_theory,
                    ratio=float(failure_evals) / support_mean_theory,
                    shrink_halved=shrink_halved,
                    coord_check=coord_check,
                )
            )
    return rows


def print_table(rows: list[ScaleRow]) -> None:
    header = (
        "D",
        "q",
        "fail_tol",
        "fail_evals",
        "E[K] MC",
        "E[K] theory",
        "evals/E[K]",
        "shrink",
        "coords",
    )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                (
                    f"{row.num_dim:,}",
                    f"{row.num_arms:,}",
                    f"{row.failure_tolerance:,}",
                    f"{row.failure_evals:,}",
                    f"{row.raasp_mean_mc:.4f}",
                    f"{row.raasp_mean_theory:.4f}",
                    f"{row.ratio:,.1f}",
                    "yes" if row.shrink_halved else "no",
                    "yes" if row.coord_check else "no",
                )
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=_parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--qs", type=_parse_int_list, default=DEFAULT_QS)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

    rows = build_rows(
        dims=args.dims,
        qs=args.qs,
        num_samples=int(args.samples),
        seed=int(args.seed),
    )
    print_table(rows)


if __name__ == "__main__":
    main()
