# Yubo: fitting_time benchmark + synthetic sine payload tests

TRIGGER: fitting_time lazy __getattr__
ADVICE: New public symbols under `analysis/fitting_time` (e.g. `fit_*`, `benchmark_enn_incremental_add_timing`, `EnnIncrementalTimingResult`) must be in `__init__.py` `__all__` and the matching `__getattr__` branch, or lazy imports raise `AttributeError`. Add `test_*_importable_from_package` for new exports.
CONFIDENCE: 3

TRIGGER: draw_benchmark_synthetic_xy RNG order
ADVICE: `draw_benchmark_synthetic_xy` draws all train `x`, then env rewards, then one `0.1 * torch.randn(N, 1)` batch—do not interleave per-row `randn(1, 1)` when matching that draw. Use `_train_xy_unit_cube_segment(..., n_train, start_row)` in `fitting_time_enn_incremental_draw.py` so checkpoint `N` matches `draw_benchmark_synthetic_xy(N=...)`.
CONFIDENCE: 3

TRIGGER: incremental ENN add timing
ADVICE: `benchmark_enn_incremental_add_timing` lives in `analysis/fitting_time/fitting_time_enn_incremental.py`; empty `EpistemicNearestNeighbors` + per-obs `add()` with `_SYNTHETIC_OBS_VAR`; fixed `ENNParams(k=min(25,max(1,N-1)),...)` at checkpoints (no `enn_fit`); M=1000 test via `draw_benchmark_test_xy_unit_cube`; grid `ENN_INCREMENTAL_CHECKPOINT_NS` / `enn_incremental_checkpoint_ns()`.
CONFIDENCE: 3

TRIGGER: fit_ind, enn_fit_ind, per-add enn_fit
ADVICE: `benchmark_enn_fit_ind_timing` in `analysis/fitting_time/fitting_time_enn_fit_ind.py`: one ENN per job; untimed per-row `add()`, then timed `enn_fit(..., num_fit_candidates=1, k` and `num_fit_samples` from `enn_fit_k_and_num_fit_samples(current_n), params_warm_start=previous ENNParams)`. Checkpoint sums → JSON vector `fit_seconds`; `log_likelihood` via `enn_test_log_likelihood` (same fixed checkpoint params as `add_method`). Artifacts: `enn_fit_ind_D{d}_{fn}_pseed...` (`experiments/modal_enn_fit_ind_batches.py`). Modal/CLI exp type `fit_ind`; job shape matches `add_method` (7 fields); `local-fit-ind` in `ops/enn_incremental_batches_local.py`.
CONFIDENCE: 0

TRIGGER: enn_fit params_warm_start, enn_fit_k
ADVICE: `enn.enn.enn_fit.enn_fit(model, *, k, num_fit_candidates, num_fit_samples, rng, params_warm_start=None)` returns `ENNParams`; `fit_ind` chains warm-start after every add. Always pass `k` from `enn_fit_k_and_num_fit_samples(current_n)` alongside warm-start (Rust gets both `k` and warm-start scales).
CONFIDENCE: 0

TRIGGER: _SYNTHETIC_OBS_VAR location
ADVICE: `_SYNTHETIC_OBS_VAR = 0.1**2` is defined in `analysis/fitting_time/fitting_time.py`, not `evaluate_metrics.py`; import from `fitting_time` (or duplicate constant only if avoiding circular imports).
CONFIDENCE: 3

TRIGGER: SURROGATE_BENCHMARK_KEYS enn_hnsw
ADVICE: `SURROGATE_BENCHMARK_KEYS` includes `enn_hnsw`; benchmark smoke tests should assert `enn_hnsw` finiteness; JSON fixtures must use `kwargs.get(k, missing)` (or `make_surrogate_benchmark`) so new keys default to NaN without `KeyError`.
CONFIDENCE: 3

TRIGGER: synthetic benchmark test helper
ADVICE: Share `_bench`/`_br` via `tests/synthetic_sine_benchmark_helpers.py` (`make_surrogate_benchmark`, `bench_result`) to satisfy kiss `VIOLATION:duplication`. Import as `make_surrogate_benchmark`/`bench_result`, not `bench`, to avoid shadowing `bench, meta = synthetic_sine_benchmark_from_payload(...)`.
CONFIDENCE: 3

TRIGGER: replace_all _bench test rename
ADVICE: Renaming `_bench(` / `_br(` in tests: replace only `_bench(` and `_br(`—never bare `_bench`/`_br` or `_benchmark` strings corrupt (e.g. `synthetic_sine_benchmark` → `synthetic_sinebenchmark`).
CONFIDENCE: 3

TRIGGER: benchmark unit cube docstring
ADVICE: `_surrogate_metric_triples_from_tensors` always maps actions with `env_action_coords_to_surrogate_unit_x` before every fit; `b_fast_only` only skips slow fits (sets NaN), not the coordinate transform. Keep `benchmark_synthetic_sine_surrogates` docstring aligned.
CONFIDENCE: 3

TRIGGER: modal benchmark rep slug meta
ADVICE: In batch tests, per-rep JSON `problem_seed` in payload must match `*_rep_json_dest(..., problem_seed=...)` slug; aggregate meta can be right while a shard meta is wrong and hide bugs.
CONFIDENCE: 3

TRIGGER: malvin review blocking
ADVICE: Read `_malvin/<run_id>/review.md` before closing; fix export gaps, slug/meta mismatches, doc drift; run `kiss check`, `ruff check .`, `PYTHONPATH=. pytest -sv tests`; write exactly `LGTM` to `review.md` only when all gates pass.
CONFIDENCE: 3

TRIGGER: full_optimization, full_opt, enn full opt
ADVICE: Fifth `enn_incremental_batches` exp type `full_optimization`: core `analysis/fitting_time/fitting_time_enn_full_opt.py`; batch keys/JSON in `experiments/modal_enn_full_opt_batches.py` + `modal_enn_full_opt_batches_json.py`. Jobs are `(env_tag, problem_seed, rep_index, num_reps, index_driver)`—not `(d, function_name, ...)`. CLI `local-full-opt ENV_TAG REP INDEX_DRIVER`. Default grid: four `DEFAULT_SYNTH_10D_ENV_TAGS` × flat/hnsw × `num_reps` (80 jobs when `index_driver=all`, `num_reps=10`).
CONFIDENCE: 0

TRIGGER: full opt checkpoint, proposal_elapsed, _cum_dt_proposing
ADVICE: After each `Optimizer.iterate()`, when `_i_iter == N` for N in `enn_incremental_checkpoint_ns()`, append `(N, Optimizer._cum_dt_proposing)`—same quantity as ITER log `proposal_elapsed`. Plan Q1 uses iteration index, not `env_steps_total`; Q5 fixed arms/denoise/policy so N iterations equals N env steps.
CONFIDENCE: 0

TRIGGER: full optimization modal, 5h worker, batch submitter
ADVICE: Modal `full_optimization` uses `enn_full_optimization_batch_worker` (5h) not the 12h `enn_incremental_batch_worker`; `enn_incremental_batch_submitter` in `modal_enn_incremental_batches_impl.py` picks worker from `experiment_type_from_tag(tag)`. Worker unit tests monkeypatch `experiments.modal_enn_incremental_batch_worker`, not impl.
CONFIDENCE: 0

TRIGGER: full_opt Q3 Q5, problem_seed rep_index
ADVICE: `benchmark_enn_full_optimization_proposal_timing` enforces Q3 (`_validate_problem_seed_for_rep_index`: `problem_seed == 18 + rep_index`) and Q5 (`_validate_q5_hyperparameters`: arms/denoise/policy fixed). Submit warns `--problem-seed` and `-d` are ignored for `full_optimization`; seeds come from `problem_seed_from_rep_index(rep_index)` in `iter_full_opt_jobs`.
CONFIDENCE: 0

TRIGGER: full_opt wall_clock, partial JSON, submitted dict
ADVICE: `SIGTERM` sets `_wall_clock_stop_requested` → `stop_reason=wall_clock_limit` and partial checkpoints in `finally`. Partial runs retry after collect: deleter removes keys from results and `submitted`; incomplete files fail `full_opt_result_json_complete` (full checkpoint grid required). Stuck in `submitted` only when worker writes no result (hard kill)—pre-existing batch pattern.
CONFIDENCE: 0

TRIGGER: full_opt eager import, fitting_time __init__
ADVICE: `EnnFullOptTimingResult` / `benchmark_enn_full_optimization_proposal_timing` are eagerly imported in `analysis/fitting_time/__init__.py` (unlike lazy `__getattr__` branches for some other fitting_time exports). New symbols still need `__all__` + tests; do not assume all fitting_time exports are lazy.
CONFIDENCE: 0
