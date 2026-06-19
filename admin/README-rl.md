# RL Environment Setup

The current project setup path is the Pixi `yubo` environment:

```bash
pixi install -e yubo
pixi run -e yubo setup
pixi run -e yubo check
```

Use `admin/README-pixi.md` for Modal and IsaacLab details.

## Legacy Micromamba Path

The shell scripts in this directory are retained for the older RL-focused
micromamba setup path on macOS and Linux.

- Shared environment from `admin/conda-rl.yml`
- Python dependencies from `requirements.txt`
- TorchRL dependencies
- BO extras (`VecchiaBO`, `ennbo`, `LassoBench`) by default

Base install:

```bash
bash admin/setup-rl.sh
```

Platform wrappers:

```bash
bash admin/setup-rl-macos.sh
bash admin/setup-rl-linux.sh
```

Useful options:

```bash
bash admin/setup-rl.sh --env-name my-env
```

Quick verification:

```bash
python -c "import torchrl, tensordict; print('torchrl ok')"
```
