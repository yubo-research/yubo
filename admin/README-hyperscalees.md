# HyperscaleES Modal Setup

## Install

Install Modal locally and authenticate once:

```bash
modal setup
```

Build the image and run preflight:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --command preflight
```

Check IsaacLab too:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --command isaaclab-preflight
```

## Image layers

```text
1. debian_slim + apt + pixi binary        # rarely changes
2. pixi.toml + pixi.lock (copied in)       # changes on dep edits
3. pixi install + setup + patchelf + check # the expensive solve/build
4. repo mount at /root                     # remounts without rebuild
```

The cached image carries OS packages, Pixi, `pixi.toml`, `pixi.lock`, and the
**hyperscalees** Pixi env. Changing `pixi.toml`, `pixi.lock`, or setup tasks
rebuilds layer 3. Source/config edits remount the repo without rebuilding Pixi.

`pixi run setup` installs vLLM, ennbo, VecchiaBO, LassoBench, mujoco_playground,
and pins numpy/numba/llvmlite last (so vLLM cannot leave numpy < 2.3).

IsaacLab configs resolve to the `isaaclab` Pixi env, installed on first use at
runtime (not baked into the image).

## Run

BO:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/bo/eggroll/paper/rl/cartpole_v1_eggroll.toml
```

UHD:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/uhd/nanochat/tinystories_d12_mezo.toml
```

LLM:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/llm/gsm8k_qwen3_1p7b_eggroll_smoke.toml
```

RL:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/rl/gymnasium/cheetah/sac_torchrl_halfcheetah_sota_like_linux.toml
```

Tests (CPU worker; pytest + pytest-testmon come from the Pixi env):

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --pytest
modal run ops/modal_hyperscalees_pixi_setup.py --pytest --pytest-args '-sv tests/test_fitting_time_enn_fit.py -rs'
```

Run detached so the job survives a local disconnect:

```bash
modal run -d ops/modal_hyperscalees_pixi_setup.py --pytest
```

## Routing

`--config` picks the runner from TOML schema:

```text
[experiment] -> ./ops/experiment.py local
[uhd]        -> ./ops/exp_uhd.py local
[llm]        -> ./ops/llm.py local
[rl]         -> ./ops/rl.py local
```

IsaacLab configs run in the `isaaclab` Pixi env. Everything else runs in
`hyperscalees`.

## Data

TinyStories nanochat configs require:

```text
data/tinystories.bin
```

Prepare it locally before running:

```bash
python scripts/prepare_tinystories.py
```

`data/` is mounted into Modal. `.pixi/`, caches, `results/`, `runs/`,
`artifacts/`, videos, checkpoints, and model weights are ignored.

## Escape Hatch

Use raw commands only when needed:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --command 'python -c "import torch; print(torch.__version__, torch.version.cuda)"'
modal run ops/modal_hyperscalees_pixi_setup.py --pixi-env isaaclab --command 'python -m problems.isaaclab_env_adapters --keyword G1'
```

## Notes

- Do not mount `.pixi/`.
- IsaacLab is bootstrapped on demand for `isaaclab` configs / `--command isaaclab-preflight` (skipped when `/opt/yubo-pixi/.isaaclab_bootstrap_ok` exists in a warm container).
- Do not switch Isaac/Torch to CUDA 13; the stack uses CUDA 12.8 wheels.
- `admin/setup-hyperscalees.sh` is the micromamba equivalent for bare-metal Linux GPUs.
