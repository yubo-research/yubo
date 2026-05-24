# HyperscaleES Modal Setup

## Install

Install Modal locally and authenticate once:

```bash
modal setup
```

Build or refresh the Modal image:

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
modal run ops/modal_hyperscalees_pixi_setup.py --config configs/rl/dm_control/quadruped/sac_puffer_quadruped_run_aggressive_short.toml
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
[rl]         -> python -m rl.runner --config
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
- Do not switch Isaac/Torch to CUDA 13; the stack uses CUDA 12.8 wheels.
- Source/config edits do not rebuild the Pixi envs.
- `admin/setup-hyperscalees.sh` is legacy local CUDA setup, not the main
  collaborator path.
