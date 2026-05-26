# RL Environment Setup

This is the RL-focused setup path for both macOS and Linux.

## What it installs

- Shared environment from `admin/conda-rl.yml`
- Python dependencies from `requirements.txt`
- TorchRL dependencies
- BO extras (`VecchiaBO`, `ennbo`, `LassoBench`) by default

## Base install (macOS or Linux)

```bash
bash admin/setup-rl.sh
```

## Platform wrappers

```bash
bash admin/setup-rl-macos.sh
bash admin/setup-rl-linux.sh
```

## Useful options

```bash
bash admin/setup-rl.sh --env-name my-env
```

## Quick verification

```bash
python -c "import torchrl, tensordict; print('torchrl ok')"
```

## HyperscaleES

HyperscaleES-backed experiments use a separate CUDA-only environment so their
JAX/LLM dependencies do not mutate `yubo-rl`.

```bash
bash admin/setup-hyperscalees.sh
```

See `admin/README-hyperscalees.md`.
