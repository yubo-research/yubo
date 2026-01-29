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
    hyperopt==0.2.7
    """.split("\n")
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        print("REQ:", req)
        sreqs.append(req)

    sreqs_2 = [
        "git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code",
        "sparse-ho @ https://github.com/QB3/sparse-ho/archive/master.zip",
        "LassoBench @ git+https://github.com/ksehic/LassoBench.git",
        "ennbo>=0.1.7",
    ]

    image = modal.Image.debian_slim(python_version="3.11.9").apt_install("swig").apt_install("git").pip_install(sreqs).pip_install(sreqs_2)

    local_enn = Path(__file__).resolve().parents[2] / "enn"
    if local_enn.exists():
        image = image.add_local_dir(str(local_enn), remote_path="/root/enn_local")
        # The shim in bbo/enn/__init__.py will find it at /root/enn_local
        image = image.env({"PYTHONPATH": "/root:/root/experiments"})
    else:
        image = image.env({"PYTHONPATH": "/root:/root/experiments"})

    project_root = Path(__file__).resolve().parents[1]
    for d in [
        "acq",
        "analysis",
        "common",
        "enn",
        "experiments",
        "model",
        "ops",
        "optimizer",
        "problems",
        "sampling",
        "third_party",
        "torch_truncnorm",
        "turbo_m_ref",
        "uhd",
    ]:
        image = image.add_local_dir(str(project_root / d), remote_path=f"/root/{d}")

    return image
