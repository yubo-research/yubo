# RL Environment Setup

This is the RL-focused setup path for both macOS and Linux.

## What it installs

- Shared environment from `admin/conda-rl.yml`
- Python dependencies from `requirements.txt`
- `pufferlib` backend (`--no-deps`) by default
- BO extras (`VecchiaBO`, `ennbo`, `LassoBench`) by default

## Base install (macOS or Linux)

```bash
bash admin/setup-rl.sh
```
On Linux, this automatically installs CUDA build tools in the env before
pufferlib. On macOS, CUDA setup is skipped.

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
micromamba run -n yubo-rl python -c "from rl.pufferlib_compat import import_pufferlib_modules; import_pufferlib_modules(); print('pufferlib ok')"
```
