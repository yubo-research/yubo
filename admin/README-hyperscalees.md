# HyperscaleES on Modal

Modal runner: `ops/modal_hyperscalees_pixi_setup.py`. Two Pixi envs in the image:
**hyperscalees** (JAX, BO, RL, pytest) and **isaaclab** (Isaac Sim + Isaac Lab, layer 4).

## Quick start

```bash
modal setup
modal run ops/modal_hyperscalees_pixi_setup.py --command preflight
modal run ops/modal_hyperscalees_pixi_setup.py --command isaaclab-preflight
```

Run an experiment:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --config <path/to/config.toml>
```

Detached (survives disconnect): add `-d` before the script path.

## Pixi env routing

| Config | Pixi env |
|--------|----------|
| Path contains `isaaclab/`, external EggRoll (`optimizer.name = "eggroll"`) | `isaaclab` |
| Path contains `isaaclab/`, JAX EggRoll (`eggroll/jax_sim=true`) | `hyperscalees` + Isaac on `PYTHONPATH` |
| Everything else | `hyperscalees` |

TOML section → runner: `[experiment]` → `./ops/experiment.py local`, `[uhd]` → `exp_uhd.py`,
`[llm]` → `llm.py`, `[rl]` → `rl.py`.

JAX Isaac configs: `configs/bo/isaaclab/g1_flat_eggroll_jax_{smoke,boost,long}.toml`.
Boost example (32 batched worlds × 128 steps):

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/bo/isaaclab/g1_flat_eggroll_jax_boost.toml
```

## Examples

```bash
# BO
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/bo/eggroll/paper/rl/cartpole_v1_eggroll.toml
# UHD / LLM / RL — same pattern, swap config path
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/uhd/nanochat/tinystories_d12_mezo.toml
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/llm/gsm8k_qwen3_1p7b_eggroll_smoke.toml
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/rl/gymnasium/cheetah/sac_torchrl_halfcheetah_sota_like_linux.toml

# Tests (CPU worker; deps from Pixi)
modal run ops/modal_hyperscalees_pixi_setup.py --pytest
modal run ops/modal_hyperscalees_pixi_setup.py --pytest --pytest-args '-sv tests/test_fitting_time_enn_fit.py -rs'
```

## Image layers

```text
1. debian_slim + pixi
2. pixi.toml + pixi.lock
3. hyperscalees install + setup + check
4. isaaclab install + check          # Isaac Sim download; cached after first build
5. repo mount at /root               # config/source edits skip rebuild
```

Edits to `pixi.toml`, `pixi.lock`, or setup tasks rebuild layers 3–4. `pixi run setup` pulls
vLLM, ennbo, VecchiaBO, LassoBench, mujoco_playground and re-pins numpy/numba last.

## Data & mounts

TinyStories nanochat needs `data/tinystories.bin` (`python scripts/prepare_tinystories.py`).
`data/` is mounted; `.pixi/`, caches, `runs/`, `artifacts/`, checkpoints are not.

## Bare-metal GPU (GCP / SSH)

Use one micromamba env instead of dual Pixi:

```bash
bash admin/setup-hyperscalees.sh
```

## Raw commands

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --command 'python -c "import torch; print(torch.__version__, torch.version.cuda)"'
modal run ops/modal_hyperscalees_pixi_setup.py --pixi-env isaaclab --command 'python -m problems.isaaclab_env_adapters --keyword G1'
```

## Constraints

- Do not mount `.pixi/`.
- Stack is CUDA 12.8 wheels — do not bump Isaac/Torch to CUDA 13.
- Isaac Lab is baked in layer 4; runtime bootstrap is a no-op when the marker exists.
- Pixi uses `kinetix-env`, not PyPI package `kinetix`.
