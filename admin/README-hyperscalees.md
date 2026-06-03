# HyperscaleES

### Local
`curl -fsSL https://pixi.sh/install.sh | bash`
```bash
pixi run setup && pixi run check && pixi run test
```

On macOS, real `llm:*` UHD configs need the opt-in Metal LLM runtime:
```bash
pixi run llm-mac && pixi run check-mac-llm
```

That path builds vLLM and `vllm-metal` from source for Apple Silicon. The default
`setup`/`check` path stays focused on the base hyperscalees BO/JAX environment.

### Remote (Modal)
```bash
modal run ops/modal_hyperscalees_pixi_setup.py --command preflight
modal run ops/modal_hyperscalees_pixi_setup.py --config <config.toml>
modal run ops/modal_hyperscalees_pixi_setup.py --pytest
```

### Routing
| Path | Env |
| :--- | :--- |
| `configs/bo/isaaclab/` | `isaaclab` |
| `eggroll/jax_sim=true` | `hyperscalees` |
| Default | `hyperscalees` |

- `[experiment]` &rarr; `experiment.py`
- `[uhd]` &rarr; `exp_uhd.py`
- `[llm]` &rarr; `llm.py`
- `[rl]` &rarr; `rl.py`

### Constraints
- CUDA 12.8 (max)
- `data/` mounted; `.pixi/` not.
- `ennbo` source-built.

---
*Bare-metal: `bash admin/setup-hyperscalees.sh`*
