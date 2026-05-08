# EggRoll CUDA Environment Setup

HyperscaleES-backed experiments and owned Qwen/vLLM LoRA experiments use a
separate CUDA-only micromamba environment from `yubo-rl`. This environment
follows the upstream ESHyperscale GPU stack and is intended for remote GPU
machines, not local CPU/macOS development.

## Install

```bash
bash admin/setup-hyperscalees.sh
```

The setup script creates `yubo-hyperscalees`, builds HyperscaleES from a
pinned git ref as a normal wheel, installs upstream-compatible CUDA JAX, and
installs the runtime packages needed by the Yubo EggRoll path. This is also the
environment for the owned Qwen/vLLM/LoRA runner behind `ops/llm.py`.
Activating the env also sets `HF_HOME`, `HF_HUB_CACHE`, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Useful options:

```bash
bash admin/setup-hyperscalees.sh --recreate-env
bash admin/setup-hyperscalees.sh --skip-verify
bash admin/setup-hyperscalees.sh --jax-spec 'jax[cuda12]==0.8.1'
```

Use `--recreate-env` when replacing an older Python 3.13/JAX-pinned env with
the current Python 3.12 CUDA env. The default JAX pin is
`jax[cuda12]==0.8.1` for the current HyperscaleES/JAX script stack. The owned
math reward path uses `math-verify[antlr4_9_3]` directly so the grader stays
compatible with HyperscaleES/Hydra's `antlr4-python3-runtime==4.9.*`
requirement without installing `gem-llm`.

The HyperscaleES wheel is repaired during setup before installation: eager
top-level imports are removed so LLM/pretrain modules do not pull RL-only
`distrax`/TFP on import, and the RWKV tokenizer files are included under the
installed `hyperscalees` package. The runtime imports `hyperscalees` normally;
there is no editable install, `.pth`, or source checkout on `PYTHONPATH`.

## Check Installed Dependencies

```bash
python - <<'PY'
import hyperscalees
import ray
import vllm
from hyperscalees.environments import llm_bandits
print(hyperscalees.__name__, len(llm_bandits.all_tasks))
print("vllm", vllm.__version__, "ray", ray.__version__)
PY
```

`HF_HOME` and `HF_HUB_CACHE` default to `~/.cache/yubo/hyperscalees` so model
and dataset cache files do not land in the repository.

## Run Yubo EggRoll Experiments

EggRoll is exposed through the normal experiment schema:

```toml
[experiment]
env_tag = "gymnax:Swimmer-misc"
policy_tag = "eggroll-ac-mlp-64x2-silu"

[optimizer]
name = "eggroll"
```

Run the smoke config with the isolated environment:

```bash
python -m experiments.experiment local configs/bo/gymnax/swimmer/eggroll_swimmer_smoke.toml
```

The policy architecture belongs in `policy_tag`. The optimizer/noiser knobs
belong in `[optimizer.params]`.

The broader coverage inventory for EggRoll paper experiments is tracked in
`docs/eggroll_experiment_coverage.md`.

## Source-Repo Boundary

`hyperscalees` is the only upstream research repo installed and imported as
Python code. Script-shaped upstream repos such as `nano-egg` and
`eggroll-vllm` are reference implementations, not clean runtime dependencies.
New Yubo runners should port the needed behavior into this repo and import only
normal Python packages plus `hyperscalees`.

## Run Owned LLM Configs

The owned LLM surface uses the same tag vocabulary as the rest of the repo:
`env_tag` for the task, `policy_tag` for the model/adaptation surface, and
`optimizer` for the update rule.

```toml
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
```

Validate the current smoke config without launching vLLM:

```bash
./ops/llm.py local configs/llm/gsm8k_qwen3_1p7b_eggroll_smoke.toml --dry-run
```

Run it from the same `yubo-hyperscalees` env on a CUDA machine:

```bash
./ops/llm.py local configs/llm/gsm8k_qwen3_1p7b_eggroll_smoke.toml
```

If the same LLM task/model is run with UHD, use the existing `[uhd]` schema and
`ops.exp_uhd`, not `[llm]`.

## Run Upstream HyperscaleES LLM Script Experiments From TOML

The clean Yubo/UHD path uses the installed `hyperscalees` package directly.
The older upstream script wrapper is only for running un-packaged upstream
`llm_experiments` scripts from an explicit source checkout. It is not required
for UHD pretraining objectives.

```bash
./ops/hyperscalees_llm.py local configs/pretrain/hyperscalees/general_do_evolution_fastzero_smoke.toml
./ops/hyperscalees_llm.py local configs/pretrain/hyperscalees/general_do_evolution_multi_gpu_fastzero_smoke.toml
./ops/hyperscalees_llm.py local configs/pretrain/hyperscalees/sft_evolution_fastzero_smoke.toml
./ops/hyperscalees_llm.py local configs/pretrain/hyperscalees/do_grpo_fastzero_smoke.toml
./ops/hyperscalees_llm.py local configs/pretrain/hyperscalees/do_grpo_multi_gpu_fastzero_smoke.toml
```

Use `--dry-run` to validate command generation without downloading models or
starting compilation:

```bash
./ops/hyperscalees_llm.py local configs/pretrain/hyperscalees/general_do_evolution_fastzero_smoke.toml --dry-run
```

The selected upstream script is declared in `[hyperscalees].script`; upstream
CLI flags belong in `[args]`, using TOML keys like `model_choice`, `num_epochs`,
and `parallel_generations_per_gpu`. These script configs require
`[experiment].repo_dir` to point at a real HyperscaleES source checkout.

## Validate EggRoll Coverage Configs

Validate the currently wired EggRoll-related TOML configs without launching
training:

```bash
python -m experiments.eggroll_coverage validate
```

Configs that point at known stale upstream `7g*` checkpoint names are reported
as `ASSET_BLOCKED` rather than invalid. Use `--require-live-assets` when a
validation run should fail on those.

## Dependency Boundary

Do not install the HyperscaleES CUDA/JAX/vLLM stack into `yubo-rl`. It pulls
broad GPU dependencies including JAX, JAXLIB, Torch, vLLM, Ray, PEFT,
datasets, transformers, reasoning-gym, and math-verify. Keeping this single
heavy stack in `yubo-hyperscalees` is the intended boundary.
