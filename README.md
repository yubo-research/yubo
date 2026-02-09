

# Installation

## Installation on MacOS

```bash
micromamba env create -n yubo -f admin/conda-macos.yml
micromamba activate yubo
pip install -r requirements.txt

ENV_LIB="${CONDA_PREFIX}/lib" LDFLAGS="-L${CONDA_PREFIX}/lib" LIBRARY_PATH="${CONDA_PREFIX}/lib" DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" CPATH=$(python -c 'import pybind11; print(pybind11.get_include())') pip install --no-build-isolation "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"

pip install "LassoBench @ git+https://github.com/ksehic/LassoBench.git" --no-deps

pip install ennbo --no-deps
cargo install kiss-ai
```

Apologies for the complexity of installation. We're blending lots of algorithms and test environments, much of which is research-quality code, some of which may be unmaintained.


## Installation on Linux

```bash
micromamba env create -n yubo -f admin/conda.yml
micromamba activate yubo
pip install -r requirements.txt

ENV_LIB="${CONDA_PREFIX}/lib" LDFLAGS="-L${CONDA_PREFIX}/lib" LIBRARY_PATH="${CONDA_PREFIX}/lib" DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" CPATH=$(python -c 'import pybind11; print(pybind11.get_include())') pip install --no-build-isolation "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"

pip install "LassoBench @ git+https://github.com/ksehic/LassoBench.git" --no-deps

pip install ennbo --no-deps
cargo install kiss-ai
```

## Setup
```bash
pre-commit install
```


## Verification

```bash
pytest -sv tests
```

## Micromamba Branch Setup (`admin/mb`)

Use these branch-specific files if you want reproducible setup flows with pinned `torch`/`torchrl`/`tensordict` behavior:

```bash
# macOS (MPS)
micromamba env create -n yubo_tmp_mb_macos -f admin/mb/conda-macos.yml

# Linux (CUDA)
micromamba env create -n yubo_tmp_mb_linux -f admin/mb/conda.yml
```

One-command full setup + verification scripts:

```bash
# from repo root
./admin/setupmacos.sh
./admin/setuplinux.sh
```

These scripts perform environment creation, dependency installation, and `pytest -sv tests -rs`.

## Branch Add-ons

Use these only if you want this branch's stricter/reproducible setup on top of upstream:

```bash
# Optional: pin ennbo instead of floating latest
pip install ennbo==0.2.1 --no-deps

# Optional: pin pyvecch commit used on this branch
pip uninstall -y pyvecch
pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall \
  "pyvecch @ git+https://github.com/feji3769/VecchiaBO.git@88fd0cc972ce07dee0f13881fd002f7412dfa617#subdirectory=code"
```

Optional add-ons in the same `yubo` env:

- `dm-control`/`mujoco`/`moviepy` are installed via `requirements.txt`.
