from pathlib import Path

import modal


# git+https://github.com/chebpy/chebpy.git
# tsroots==0.1.22
def mk_image():
    reqs = """
    numpy==1.26.4
    scipy==1.15.3
    torch==2.3.1
    botorch==0.12.0
    gpytorch==1.13
    gymnasium==1.2.0
    cma==4.0.0
    mujoco==3.3.3
    optuna==4.0.0
    gymnasium[box2d]
    gymnasium[mujoco]
    faiss-cpu==1.9.0
    nds==0.4.3
    matplotlib==3.10.8
    celer==0.7.4
    numba==0.60.0
    hyperopt==0.2.7
    smac==2.3.1
    maturin>=1.0
    """.split("\n")
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        sreqs.append(req)

    sreqs_2 = [
        "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code",
        "sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip",
        "LassoBench @ git+https://github.com/ksehic/LassoBench.git",
    ]

    image = (
        modal.Image.debian_slim(python_version="3.11.9")
        .apt_install("swig", "git", "gcc", "g++", "curl", "build-essential")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            'echo "export PATH=$HOME/.cargo/bin:$PATH" >> ~/.bashrc',
        )
        .pip_install(sreqs)
        .pip_install(sreqs_2, extra_options="--no-deps")
    )

    image = image.env({"PYTHONPATH": "/root"})

    project_root = Path(__file__).resolve().parents[1]
    enn_root = project_root.parents[0] / "enn"

    # Patterns to exclude when copying enn (build artifacts, caches, git)
    enn_ignore = [
        ".git",
        "target",
        "**/debug",
        "**/release",
        "**/__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "*.egg-info",
        ".mypy_cache",
        ".venv",
        "**/*.whl",
    ]

    # Add the full enn project and build the Rust extension
    # copy=True required because we run build commands after adding local files
    image = image.add_local_dir(str(enn_root), remote_path="/root/enn", ignore=enn_ignore, copy=True)
    image = image.run_commands(
        ". $HOME/.cargo/env && cd /root/enn/rust/crates/ennbo-py && maturin build --release",
        "pip install $(find /root/enn/rust -path '*/wheels/*manylinux*.whl' | head -1) && pip install -e /root/enn",
    )
    image = image.run_commands(
        "python -c \"from enn.enn.enn_fit import enn_fit; print('enn import OK')\"",
    )

    for d in [
        "acq",
        "analysis",
        "common",
        "experiments",
        "model",
        "ops",
        "optimizer",
        "rl",
        "problems",
        "sampling",
        "torch_truncnorm",
        "turbo_m_ref",
    ]:
        image = image.add_local_dir(str(project_root / d), remote_path=f"/root/{d}")

    return image
