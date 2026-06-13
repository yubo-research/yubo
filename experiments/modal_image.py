from pathlib import Path

import modal


# git+https://github.com/chebpy/chebpy.git
# tsroots==0.1.22
def mk_image(tag: str = "default"):
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
        .apt_install(
            "swig",
            "git",
            "gcc",
            "g++",
            "curl",
            "build-essential",
            # ndarray-linalg / ennbo link cblas; resolve at build + runtime (e.g. cblas_ddot).
            "libopenblas-dev",
            # ennbo uses faiss-rs `static` feature: vendored Faiss is built with cmake (needs C API for bindings).
            "cmake",
            "ninja-build",
            # maturin vendors DT_NEEDED libs into the wheel via patchelf when linking openblas.
            "patchelf",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            'echo "export PATH=$HOME/.cargo/bin:$PATH" >> ~/.bashrc',
        )
        .pip_install(sreqs)
        # VecchiaBO's setup imports torch during get_requires_for_build_wheel; pip's
        # isolated build env does not see the prior layer — disable isolation.
        .pip_install(sreqs_2, extra_options="--no-deps --no-build-isolation")
    )

    image = image.env({"PYTHONPATH": "/root"})

    project_root = Path(__file__).resolve().parents[1]
    enn_root = project_root.parents[0] / "enn"

    from ops.modal_enn_image import add_enn_to_image

    # Add the full enn project and build the Rust extension
    image = add_enn_to_image(image, enn_root)
    image = image.run_commands(
        "python -c \"from enn.enn.enn_class import EpistemicNearestNeighbors; print('enn import OK')\"",
    )

    for d in [
        "acq",
        "analysis",
        "common",
        "experiments",
        "model",
        "ops",
        "optimizer",
        "policies",
        "problems",
        "rl",
        "sampling",
        "torch_truncnorm",
        "turbo_m_ref",
    ]:
        image = image.add_local_dir(str(project_root / d), remote_path=f"/root/{d}")

    return image
