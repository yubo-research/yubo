# Pixi Setup

## Local

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install -e yubo
pixi run -e yubo setup
pixi run -e yubo check
pixi run -e yubo python -m pytest --no-testmon -sv tests -rs
```

The `yubo` Pixi environment is the primary project environment. The manifest
owns the ordered setup tasks, including upstream HyperscaleES, `ennbo`,
MuJoCo assets, LLM runtime pieces, and runtime pin repair.

IsaacLab uses a separate environment because Isaac Sim has a large pinned stack:

```bash
pixi install -e isaaclab
pixi run -e isaaclab setup
pixi run -e isaaclab check
```

On macOS, `pixi run -e yubo setup` also prepares the Metal LLM runtime used by
real `llm:*` UHD configs. That path may build vLLM and `vllm-metal` from source
for Apple Silicon.

## Modal

```bash
modal run ops/modal_pixi_setup.py
modal run ops/modal_pixi_setup.py --config <config.toml>
modal run ops/modal_pixi_setup.py --pytest --pytest-args "--no-testmon -sv tests -rs"
```

Modal pytest runs use the GPU runner by default and set `JAX_PLATFORMS=cuda,cpu`.
Use the CPU worker only when that is intentional:

```bash
modal run ops/modal_pixi_setup.py --pytest --pytest-cpu --pytest-args "--no-testmon -sv tests -rs"
```

## Routing

| Path | Env |
| :--- | :--- |
| `configs/bo/isaaclab/` | `isaaclab` |
| `eggroll/jax_sim=true` | `yubo` |
| Default | `yubo` |

- `[experiment]` -> `experiment.py`
- `[uhd]` -> `exp_uhd.py`
- `[llm]` -> `llm.py`
- `[rl]` -> `rl.py`

## Constraints

- CUDA 12.8 max in the `yubo` env.
- Modal mounts project files into `/root`; `.pixi/` is built into the image.
- `ennbo` is built from `yubo-research/enn` and patched during Pixi setup.
