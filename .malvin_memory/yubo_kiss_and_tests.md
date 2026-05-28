# Yubo: kiss + tests (detail)

TRIGGER: kiss limits, lines_per_file
ADVICE: Kiss Python `lines_per_file` threshold was 500 in session output; split files before crossing it.
CONFIDENCE: 3

TRIGGER: enn incremental kiss split, batch_worker
ADVICE: If `ops/enn_incremental_batches.py` or `experiments/modal_enn_incremental_batches_impl.py` hit `lines_per_file`, extract local CLI to `ops/enn_incremental_batches_local.py` (`register_local_commands`) and Modal worker body to `experiments/modal_enn_incremental_batch_worker.py` (`dispatch_enn_incremental_batch_worker`); keep thin `@app.function` wrapper in impl.
CONFIDENCE: 0

TRIGGER: kiss branches, _JOB_HANDLERS, dispatch_enn_incremental
ADVICE: If `dispatch_enn_incremental_batch_worker` exceeds kiss `branches_per_function` (limit 9), refactor to `_JOB_HANDLERS` plus per-exp `_handle_*` helpers (each ≤9 branches). Uniform handler signatures may need `del job_key, result_to_payload`; refactor affects all experiment types in the worker file, not only the new one.
CONFIDENCE: 0

TRIGGER: incremental batch worker test, monkeypatch worker_mod
ADVICE: Worker unit tests must `monkeypatch.setattr` on `experiments.modal_enn_incremental_batch_worker` (e.g. `benchmark_enn_fit_ind_timing`), not `modal_enn_incremental_batches_impl`, after the worker dispatch split.
CONFIDENCE: 0

TRIGGER: statements_per_file
ADVICE: Kiss Python `statements_per_file` threshold was 300 (statements inside function/method bodies in that file); split or thin hot paths.
CONFIDENCE: 3

TRIGGER: calls_per_function
ADVICE: Large setup tests can exceed `calls_per_function` (e.g. 91); extract helpers in the same package to reduce call count in one function body.
CONFIDENCE: 3

TRIGGER: local_variables_per_function
ADVICE: Group setup with `SimpleNamespace`, small builder functions, or a dedicated `*_bundle.py` helper module to stay under the local-variable cap.
CONFIDENCE: 3

TRIGGER: kiss show-tests
ADVICE: Use `kiss show-tests [FILE ...]` to iterate pytest subset while fixing a file; still run full `pytest -sv tests` before done.
CONFIDENCE: 3

TRIGGER: test_kiss_coverage_static_bridge
ADVICE: Monolith was split into `tests/test_kiss_coverage_static_bridge_a.py`, `_b.py`, `_c.py` after removing ClassDefs via `type()` / `SimpleNamespace` patterns.
CONFIDENCE: 3

TRIGGER: smoke_extra, kiss_cov_all, optimizer_modules, experiment_sampler, uhd_config
ADVICE: Same session pattern: multiple `test_*` modules replacing one huge file; some areas used dedicated stub/support modules under `tests/`.
CONFIDENCE: 2

TRIGGER: conftest sys.path
ADVICE: `tests/conftest.py` appends `tests/` to `sys.path` when absent; do not prepend—repo-root resolution must win for imports like `ops.*`.
CONFIDENCE: 3

TRIGGER: pre-commit kiss
ADVICE: If `.pre-commit-config.yaml` runs `kiss check`, local `pre-commit run --all-files` should pass alongside manual `kiss check`.
CONFIDENCE: 3

TRIGGER: Puffer SAC build_env_setup stub
ADVICE: Stubbing SAC `build_env_setup` in tests requires patching `rl.pufferlib.offpolicy.env_utils.build_continuous_gym_env_setup` (the binding used inside offpolicy `build_env_setup`), not `rl.pufferlib.sac.env_utils.build_continuous_gym_env_setup`.
CONFIDENCE: 3

TRIGGER: sac eval_utils lazy getattr
ADVICE: Tests that patch `rl.pufferlib.sac.eval_utils.<attr>` can leave names on the module dict and shadow `__getattr__` for later tests; pop those keys from `rl.pufferlib.sac.eval_utils.__dict__` before assertions or patch underlying modules (`rl.core.episode_rollout`, `rl.eval_noise`, offpolicy eval helpers).
CONFIDENCE: 2

TRIGGER: kiss duplication enabled
ADVICE: After `test_coverage` passes, `kiss check` may still fail on `VIOLATION:duplication` (e.g. min_similarity 0.7); consolidate duplicated bodies across prod and `tests/` or use shared helpers—same discipline as kiss size limits.
CONFIDENCE: 3
