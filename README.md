

# Installation

## Installation on MacOS

```bash
micromamba env create -n yubo -f admin/conda-macos.yml
micromamba activate yubo
pip install -r requirements.txt

ENV_LIB="${CONDA_PREFIX}/lib" LDFLAGS="-L${CONDA_PREFIX}/lib" LIBRARY_PATH="${CONDA_PREFIX}/lib" DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" CPATH=$(python -c 'import pybind11; print(pybind11.get_include())') pip install --no-build-isolation "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"

pip install "LassoBench @ git+https://github.com/ksehic/LassoBench.git" --no-deps

pip install ennbo --no-deps
cargo install kiss-ai --version 0.2.4
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

If your code crashes or hangs, try this [hack](https://discuss.pytorch.org/t/ran-into-this-issue-while-executing/101460):
```
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```
I don't recommend this, however, as it may slow things down.
