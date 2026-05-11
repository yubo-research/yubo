# EggRoll Experiment Coverage Inventory

This document is the working inventory for covering the EggRoll/HyperscaleES
experiment surface from Yubo without patching upstream.

## Boundary

Treat upstream as pinned runtime infrastructure:

- `hyperscalees`, installed from a pinned git ref as a normal Python dependency

`hyperscalees` is the only upstream research repo that clean Yubo runtime code
should import. Script-shaped paper repos such as `nano-egg` and
`eggroll-vllm` are reference implementations until their behavior is ported
into Yubo-owned modules. Do not add `.pth` files, `sys.path` edits, or imports
from source checkouts.

Yubo owns setup, dependency pins, tags, configs, metadata, launchers, and thin
adapters around stable upstream primitives.

The integration surface stays tag-based:

```toml
[experiment]
env_tag = "..."
policy_tag = "..."

[optimizer]
name = "eggroll"
```

Interpretation:

- `env_tag`: problem, evaluator, simulator, dataset, or verifier.
- `policy_tag`: model, policy, candidate generator, and adaptation surface.
- `optimizer.name = "eggroll"`: HyperscaleES noiser/update path and its params.

## Current Support

| Area | Current state | Evidence |
| --- | --- | --- |
| CUDA env | Supported | `admin/setup-hyperscalees.sh`, `admin/conda-hyperscalees.yml` |
| HyperscaleES package imports | Supported | `problems/pre_obj.py` imports `hyperscalees.*` directly; no `.pth` or repo-root path injection |
| Owned LLM config surface | Initial port | `ops/llm.py`, `experiments/llm.py`, `llm/registry.py`, `llm/tasks.py`, `configs/llm/gsm8k_qwen3_1p7b_eggroll_smoke.toml` |
| UHD text objective | Runtime-gated | `problems/text_obj.py`, `llm/engine_pool.py`, `configs/uhd/text/gsm8k_qwen3_1p7b_mezo.toml` |
| Upstream LLM TOML wrapper | Legacy source-checkout path | `experiments/hyperscalees_llm.py`, `configs/pretrain/hyperscalees/*.toml`; requires an explicit HyperscaleES source checkout because upstream `llm_experiments` is not packaged |
| nano-egg TOML wrapper | Legacy source-checkout path | `./ops/nanoegg_pretrain.py`, `experiments/nanoegg_pretrain.py`, `configs/pretrain/nanoegg/*.toml`; requires an explicit `nano-egg` source checkout |
| EggRoll config validation | Supported | `python -m experiments.eggroll_coverage validate` |
| Gymnax EggRoll smoke | Supported | `optimizer/eggroll_designer.py`, `policies/eggroll_policy.py`, `configs/bo/gymnax/swimmer/eggroll_swimmer_smoke.toml` |
| Full EggRoll paper coverage | Config-covered, launch-checked | Paper-intent TOMLs exist; adapter-backed families validate through `experiments.eggroll_coverage` |

## Coverage Matrix

| Experiment family | Paper surface | Best Yubo route | Status | Work needed |
| --- | --- | --- | --- | --- |
| Speed microbenchmark | Linear model, large dimension, bf16, EggRoll vs PPO/OpenES throughput | `env_tag = "synthetic:linear-speed"` plus EggRoll objective adapter | Adapter wired | Replace surrogate scoring with exact paper throughput measurement if needed |
| Integer LM pretraining | nano-egg-style int8 EGG on MiniPile | Standard `[experiment]` EggRoll route plus source-script runner for exact NanoEgg paper runs; Yubo-owned JAX NanoEgg UHD objective with task in `env_tag` and model in `policy_tag` for UHD variants | EggRoll route, source runner, and UHD route added | Add parity harness against the reference script |
| Integer pretraining ablations | Population size and data batch size sweeps | Same Yubo-owned NanoEgg UHD objective | Reference configs added | Port paper batch-16 population grid under the owned runner |
| RWKV LLM evolution | Countdown/GSM8K/math with `general_do_evolution` | Package-backed UHD LLM objective for clean runs; optional source-script wrapper for exact upstream script runs | Config added, asset gated | Live `7w*` substitutes and exact `7g*` asset-blocked configs exist |
| RWKV GRPO baseline | `do_grpo`, `do_grpo_multi_gpu` | Optional source-script wrapper | Config added, asset gated | Live `7w*` substitutes and exact `7g*` asset-blocked configs exist |
| RWKV SFT/evolution | `sft_evolution` | Optional source-script wrapper | Partial | Smoke config exists; add real configs |
| RWKV int8 distillation | Quantized RWKV distillation with KL fitness | Yubo external-objective adapter surface | Adapter wired | Replace surrogate scoring with exact KL evaluator when the model/data assets are pinned |
| Qwen RLVR | Transformer RLVR on DeepScaleR/math evals | Owned text objective under `llm:*`; vLLM backend | UHD route added, task coverage started | Add exact RLVR/eval config and verifier parity |
| Pass@k optimization | Qwen pass@k objective | Same owned text objective under `llm:*` | UHD route added, task coverage started | Add exact pass@k config and verifier parity |
| Tabula-rasa single-agent RL | CartPole, Pendulum, Brax, Craftax, Jumanji, Kinetix, Navix | Tag adapters using HyperscaleES noisers and policy models | Adapter wired | Optional runtime packages must be installed for non-Gymnax families |
| Multi-agent RL | JaxMARL/MPE-style `Simple Spread`, `Simple Speaker Listener`, `Simple Reference` | `env_tag = "jaxmarl:<name>"`, multi-agent policy tag | Adapter wired | Optional JaxMARL package must be installed |
| HFT/time-series | LOBS5-360M, Jax-LOB, PnL objective, LoRA rank 4 | `env_tag = "jaxlob:<dataset/task>"`, `policy_tag = "lobs5-360m-lora-r4"` | Adapter wired | Replace surrogate scoring with exact Jax-LOB/LOBS5 evaluator when assets are pinned |
| Toy MLP test | Upstream `tests/end_to_end_test.py` | Direct upstream test | Supported as smoke only | Keep as sanity check, not paper coverage |

## RL Adapter Detail

Current Yubo RL support is intentionally narrow:

```toml
[experiment]
env_tag = "gymnax:Swimmer-misc"
policy_tag = "eggroll-ac-mlp-64x2-silu"

[optimizer]
name = "eggroll"
```

The designer uses `hyperscalees.models.rl.ActorCriticMLP` and
`hyperscalees.noiser` through adapter-backed JAX rollouts. It supports
continuous Gymnax/Brax-style actions, discrete Gymnax/Jumanji-style actions,
JaxMARL joint-action adaptation, and one-step external-objective adapters.

Paper RL coverage needs these env-tag families:

- `gymnax:<env>` for Gymnax-compatible classic control/smoke coverage.
- `brax:<env>` for Brax continuous-control tasks.
- `craftax:<env>` for Craftax Classic/Symbolic.
- `jumanji:<env>` for 2048/Knapsack/Snake.
- `kinetix:<env>` for Kinetix tasks.
- `navix:<env>` for DoorKey/DynamicObstacles/FourRooms.
- `jaxmarl:<env>` for multi-agent tasks.

Policy tags should stay architecture-oriented:

- `eggroll-ac-mlp-256x3-silu`
- `eggroll-ac-mlp-64x3-silu`
- future multi-agent variants such as `eggroll-marl-mlp-64x3-silu`

## LLM Adapter Detail

Owned Qwen/vLLM LLM configs use one strict `[llm]` section:

```toml
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
```

The same `env_tag`/`policy_tag` pair can be used under `[uhd]` for text UHD:

```toml
[uhd]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "mezo"
text_search_dim = 128
```

This routes through `problems/text_obj.py`. The problem adapter is named for the
objective surface, not the backend: text generation and scoring. vLLM remains
the local inference backend under `llm/`.

The clean Yubo/UHD route imports the installed `hyperscalees` package directly
from `problems/pre_obj.py`. It should behave like any other dependency import:
no `.pth`, no `sys.path` edits, and no repository-root module assumptions.

The upstream script wrapper below is separate. It exists only for exact
upstream script compatibility, because upstream `llm_experiments` is not part
of the packaged `hyperscalees` import surface.

Source-checkout upstream scripts currently available:

- `llm_experiments.general_do_evolution`
- `llm_experiments.general_do_evolution_multi_gpu`
- `llm_experiments.sft_evolution`
- `llm_experiments.do_grpo`
- `llm_experiments.do_grpo_multi_gpu`

Optional source-script wrapper:

- `experiments/hyperscalees_llm.py`

Paper-intent configs currently live under:

- `configs/pretrain/hyperscalees/paper/`
- `configs/pretrain/hyperscalees/paper/exact_asset_blocked/`
- `configs/pretrain/nanoegg/paper/`
- `configs/bo/eggroll/paper/`

Reference-only upstream repos:

- `ESHyperscale/nano-egg` for integer LM pretraining.
- `ESHyperscale/eggroll-vllm` for Qwen/vLLM LoRA EggRoll experiments.

These should be read as behavior references, not imported as Python packages.

Important known issue:

- Upstream CLI accepts `7g*` model choices, but the pinned upstream model
  registry points at stale Hugging Face filenames for the `7g*` checkpoints.
  Live `7w*` and `7n*` choices work. Exact paper `7g*` runs need live assets
  via refreshed upstream registry, local cache preseed, or an explicit model
  path option if upstream supports one.

## Execution Priority

1. Harden the package-backed HyperscaleES UHD objective for real runs.
2. Add parity checks for NanoEgg MiniPile behavior against the reference script.
3. Split current Gymnax EggRoll path into reusable env adapter boundaries.
4. Add RL env families in this order: Brax, Jumanji, Navix, Craftax, Kinetix.
5. Add JaxMARL multi-agent coverage.
6. Add HFT/Jax-LOB coverage.
7. Add distillation, Qwen RLVR, and pass@k coverage after identifying the
   stable model/evaluator surfaces.

## Validation

Run config validation without launching training:

```bash
python -m experiments.eggroll_coverage validate
```

Known-stale upstream `7g*` checkpoint configs are reported as `ASSET_BLOCKED`.
Use `--require-live-assets` to make those fail validation.

## Definition Of Covered

An experiment family is considered covered when all of these exist:

1. A tag-level config or direct upstream TOML config.
2. A reproducible setup path in a clearly separate Python environment.
3. A smoke run that completes on the remote machine.
4. A paper-scale config that dry-runs to the expected command/evaluator.
5. Logs and metadata written outside the repository tree.
