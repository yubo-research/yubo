import modal


def mk_image():
    reqs = """
    numpy==1.26.0
    scipy==1.11.3
    torch==2.5.0
    botorch==0.12.0
    gpytorch==1.13
    gymnasium==1.0.0
    cma==4.0.0
    mujoco==3.2.4
    optuna==4.0.0
    gymnasium[box2d]
    gymnasium[mujoco]
    tsroots==0.1.21
    git+https://github.com/chebpy/chebpy.git
    """.split("\n")
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        print("REQ:", req)
        sreqs.append(req)
    # print("SREQS:", sreqs)
    return modal.Image.debian_slim(python_version="3.11.5").apt_install("swig").apt_install("git").pip_install(sreqs)
