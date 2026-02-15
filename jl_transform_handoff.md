# Standalone JL Transform Task Handoff

## Objective
Implement and test a standalone sparse JL embedding utility for high-dimensional parameter vectors.

This is intentionally independent of optimizer integration. Do not wire it into `UHDMeZO` yet.

## Context
- Target future use: embed high-dimensional model parameter vectors (for example, ~400k dimensions) into a lower-dimensional space (`d`, for example `100`) for surrogate modeling.
- We want deterministic, seed-driven behavior so candidate generation can be replayed exactly.

## Required Deliverables
1. A standalone JL transform module under `sampling/` (Torch preferred).
2. Unit tests under `tests/` validating correctness and practical properties.
3. No changes to optimizer integration paths (`optimizer/uhd_mezo.py`, `optimizer/uhd_loop.py`, etc.).

## Functional Requirements
- Deterministic given `(x, d, s, seed)`.
- Input vector is 1-D.
- Output shape is `(d,)`.
- Supports sparse-style projection behavior (each input coordinate maps to `s` rows with random signs).
- Raises clear error when `s > d`.
- Zero vector maps to zero vector.
- Linearity check: `T(x + z) == T(x) + T(z)` (within tolerance where appropriate).

## Suggested API
- `block_sparse_jl_transform_t(x: torch.Tensor, d: int, s: int = 4, seed: int = 42) -> torch.Tensor`

If helpful, a NumPy parity version can exist, but focus on Torch path.

## Test Expectations
Add focused tests that cover:
- determinism by seed,
- error handling (`s > d`),
- zero mapping,
- linearity,
- without-replacement style behavior per input coordinate (exactly `s` nonzero target contributions when probing basis vectors),
- optional neighbor/correlation preservation sanity check in synthetic data.

## Constraints
- Do not integrate with existing optimizer ask/tell flow yet.
- Keep implementation clear and lightweight.
- Keep changes scoped to JL utility + tests.

## Validation Commands
Run and report results for:
- `pytest -sv tests/test_sparse_jl_t.py tests/test_delta_sparse_jl_t.py`
- `ruff check`
- `kiss check`

Also run project suite if feasible:
- `pytest -sv tests`

## Definition of Done
- Standalone JL utility exists and is tested.
- Required validation commands pass.
- No optimizer integration code changed.
