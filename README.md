# Yubo

## Installation

Install Pixi, then create and check the default environment:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi run setup
pixi run check
```

The default Pixi environment installs this repository as an editable Python
package and prepares the runtime pieces used by the project.

IsaacLab uses a separate Linux/CUDA environment:

```bash
pixi install -e isaaclab
pixi run -e isaaclab setup
pixi run -e isaaclab check
```

## Development

```bash
pre-commit install
pre-commit run
pixi run python -m pytest --no-testmon -sv tests -rs
```

## Examples

```bash
pixi run ./ops/experiment.py local configs/demo/turbo_enn.toml
pixi run ./ops/experiment.py local configs/demo/ppo.toml
pixi run ./ops/exp_uhd.py local configs/demo/mezo.toml
```
