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
    """.split("\n")
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        print("REQ:", req)
        sreqs.append(req)

    sreqs_2 = ["git+https://github.com/feji3769/VecchiaBO.git#subdirectory=code"]
    # print("SREQS:", sreqs)
    return modal.Image.debian_slim(python_version="3.11.9").apt_install("swig").apt_install("git").pip_install(sreqs).pip_install(sreqs_2)
