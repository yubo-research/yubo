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
    smac==2.3.1
    ennbo==0.2.1
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
        .apt_install("swig", "git", "gcc", "g++")
        .pip_install(sreqs)
        .pip_install(sreqs_2, extra_options="--no-deps")
    )

    image = image.env({"PYTHONPATH": "/root"})
    # GitHub ``enn`` 0.3.x editable install does not ship ``enn._rust``; PyPI ``ennbo`` 0.2.1 matches
    # ``analysis.fitting_time.fit_enn`` (pure-Python ``enn_fit``).
    image = image.run_commands(
        "python -c \"from enn.enn.enn_fit import enn_fit; print('enn import OK')\"",
    )

    project_root = Path(__file__).resolve().parents[1]
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
