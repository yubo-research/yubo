

# Installation

## Installation on MacOS

```bash
micromamba env create -n yubo -f admin/conda-macos.yml
micromamba activate yubo
pip install -r requirements.txt

ENV_LIB="${CONDA_PREFIX}/lib" LDFLAGS="-L${CONDA_PREFIX}/lib" LIBRARY_PATH="${CONDA_PREFIX}/lib" DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" CPATH=$(python -c 'import pybind11; print(pybind11.get_include())') pip install --no-build-isolation "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"

pip install "LassoBench @ git+https://github.com/ksehic/LassoBench.git" --no-deps

pip install "git+https://github.com/yubo-research/enn.git" --no-deps
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

pip install "git+https://github.com/yubo-research/enn.git" --no-deps
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

## RL runs

```bash
python -m rl.algos.runner --config experiments/configs/pendulum_ppo.toml
python -m rl.algos.runner --config experiments/configs/pendulum_ppo.toml --seeds 1,2,3 --workers 3
```

You can also set this in the TOML:
```toml
[rl.run]
seeds = [1, 2, 3]
workers = 3
```

Checkpointing (optional):
```toml
[rl.ppo]
checkpoint_interval = 10
resume_from = "_tmp/pendulum_ppo/checkpoint_last.pt"
```

BO checkpoints (optional):
```toml
[experiment]
checkpoint_every = 10
resume = true
```
