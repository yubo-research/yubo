# Malvin memory index (navigation)

- `style.md` — primary TRIGGER list for this repo.
- `yubo_kiss_and_tests.md` — kiss limits, pytest layout, Puffer/SAC test-stub targets (includes session tail).
- `yubo_ppo_bo_designers.md` — BO `ppo-ac` / `ppo-pg` designers, merge/normalize pitfalls, registry layout.
- `yubo_fitting_time_benchmark.md` — `enn_hnsw`, synthetic benchmark helpers, modal rep JSON meta, lazy exports, incremental ENN `add()` / `fit_ind` / `full_optimization` timing, `enn_fit` warm-start, `draw_benchmark_synthetic_xy` RNG order.
- `yubo_modal_enn.md` — `enn_incremental_batches deploy`, `modal_image.mk_image`, sibling `../enn`, sccache/Cargo wrapper vs Modal, testmon vs full pytest.
- `yubo_turbo_enn_designers.md` — turbo-enn registry layout, HNSW `idx`/`index_driver`, `DesignerDef` vs simple table, kiss import-cycle constraint.

## Session learnings (tidy / quality gates)

TRIGGER: sac eval_utils lazy getattr
ADVICE: `rl.pufferlib.sac.eval_utils` exposes many symbols via `__getattr__`. `monkeypatch.setattr("rl.pufferlib.sac.eval_utils.<name>", ...)` injects into the module dict and can shadow lazy resolution for later tests; pop those keys from `rl.pufferlib.sac.eval_utils.__dict__` at the start of sensitive tests, or patch `rl.core.episode_rollout` / `rl.eval_noise` instead of the facade.
CONFIDENCE: 2

TRIGGER: kiss check duplication gate
ADVICE: With `duplication_enabled=true` in `.kissconfig`, a green `test_coverage` gate can still be followed by many `VIOLATION:duplication` lines; fix with shared helpers, thin re-exports, or deduped tests—not only by adding kiss coverage tests.
CONFIDENCE: 3

TRIGGER: infer_observation_spec pixel channels
ADVICE: `_infer_channels` in `rl/pufferlib/offpolicy/env_utils.py` only recognizes channel axis sizes in `{1,3,4}` (plus a 2D+fallback case); otherwise it defaults to `3`, which can disagree with `(C,H,W)` stacks (e.g. `C=2`) and break `prepare_obs_np` or mis-size policies. See `bug.md`; prefer explicit channel/layout from env or config when stacks are not 1/3/4.
CONFIDENCE: 2

TRIGGER: Cursor grep ripgrep shell
ADVICE: If workspace `grep`/`rg` tools error with `IO error for operation on : No such file`, run `cd /Users/dsweet2/Projects/yubo && rg ...` (absolute repo path) in the shell.
CONFIDENCE: 3

TRIGGER: malvin KPOP log path
ADVICE: Malvin append-only KPOP steps live under `_malvin/<run_id>/_kpop/exp_log_<run_id>.md` next to `request.md`; pair each hypothesis with a falsification run or file read.
CONFIDENCE: 3

TRIGGER: malvin review scope, review_prep, LGTM
ADVICE: Post-impl malvin review scope is plan fidelity + quality-gate blockers only. Strip plan-consistent design, test gaps, duplication, pre-existing Modal batch patterns, and unconfirmed risks from `review_prep.md` before `review.md`. Write exactly `LGTM` if nothing remains; add failing regression tests only for confirmed in-scope bugs (e.g. plan Q5/Q3 validation), not for local-debug knobs or test-coverage gaps.
CONFIDENCE: 3

TRIGGER: Puffer SAC build_env_setup stub
ADVICE: `rl.pufferlib.sac.env_utils.build_env_setup` delegates to `rl.pufferlib.offpolicy.env_utils.build_env_setup`, which calls `build_continuous_gym_env_setup` from the name bound in `rl.pufferlib.offpolicy.env_utils` (from `rl.core.env_setup`). Patch `rl.pufferlib.offpolicy.env_utils.build_continuous_gym_env_setup`, not `rl.pufferlib.sac.env_utils.build_continuous_gym_env_setup`.
CONFIDENCE: 3
