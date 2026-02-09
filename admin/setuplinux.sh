#!/usr/bin/env bash
set -euo pipefail

cd /Users/mehulbafna/Desktop/DS/bayesellipse/yubo

micromamba env remove -n yubo_tmp_mb_linux -y || true
micromamba env create -n yubo_tmp_mb_linux -f admin/mb/conda.yml

eval "$(micromamba shell hook --shell bash)"
micromamba activate yubo_tmp_mb_linux

pip install -r requirements.txt
ENV_LIB="${CONDA_PREFIX}/lib" \
LDFLAGS="-L${CONDA_PREFIX}/lib" \
LIBRARY_PATH="${CONDA_PREFIX}/lib" \
DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
CPATH="$(python -c 'import pybind11; print(pybind11.get_include())')" \
pip install --no-build-isolation "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"
pip install "LassoBench @ git+https://github.com/ksehic/LassoBench.git" --no-deps
pip install ennbo --no-deps
cargo install kiss-ai

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pytest -sv tests -rs
