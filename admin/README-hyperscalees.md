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

The image is built from:

```text
pixi.toml
pixi.lock
ops/modal_hyperscalees_pixi_base_image.py
ops/modal_hyperscalees_pixi_image.py
```

The image carries OS packages, Pixi, `pixi.toml`, `pixi.lock`, and both Pixi
envs. Changing `pixi.toml`, `pixi.lock`, or setup tasks rebuilds the env image.
Source/config edits do not rebuild the Pixi envs.

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

Tests:

```bash
modal run ops/modal_hyperscalees_pixi_setup.py --pytest
modal run ops/modal_hyperscalees_pixi_setup.py --pytest --pytest-args '-sv tests/test_experiment_sampler_runtime.py -rs'
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
- `hyperscalees` setup installs vLLM, VecchiaBO, and LassoBench.
- Do not switch Isaac/Torch to CUDA 13; the stack uses CUDA 12.8 wheels.
- `admin/setup-hyperscalees.sh` is legacy local CUDA setup, not the main
  collaborator path.
