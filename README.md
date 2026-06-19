# Installation

## Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install -e yubo
pixi run -e yubo setup
pixi run -e yubo check
```

The Pixi manifest owns both the environment and the ordered setup tasks. The
`setup` task builds and patches runtime pieces such as `ennbo`, prepares MuJoCo
assets, and restores runtime pins in the order needed by the project.

IsaacLab is a separate Pixi environment:

```bash
pixi install -e isaaclab
pixi run -e isaaclab setup
pixi run -e isaaclab check
```

## Setup

```bash
pre-commit install
```

## Verification

```bash
pre-commit run
pixi run -e yubo python -m pytest --no-testmon -sv tests -rs
```

For Modal setup and test runs:

```bash
modal run ops/modal_pixi_setup.py
modal run ops/modal_pixi_setup.py --pytest --pytest-args "--no-testmon -sv tests -rs"
```

Modal pytest runs use the GPU runner by default with `JAX_PLATFORMS=cuda,cpu`.
Use `--pytest-cpu` only when intentionally running the suite on a CPU worker.

If your code crashes or hangs, try this [hack](https://discuss.pytorch.org/t/ran-into-this-issue-while-executing/101460):

```
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

I don't recommend this, however, as it may slow things down.

---

## Examples

From the repository root (with `PYTHONPATH` set to the repo so imports resolve):

```bash
pixi run -e yubo ./ops/experiment.py local configs/demo/turbo_enn.toml
pixi run -e yubo ./ops/experiment.py local configs/demo/ppo.toml
pixi run -e yubo ./ops/exp_uhd.py local configs/demo/mezo.toml
```
