import modal


def mk_image():
    reqs = """
    numpy==1.26.0
    scipy==1.13.1
    torch==2.1.0
    botorch==0.9.2
    gpytorch==1.11
    gymnasium==0.29.1
    cma==3.3.0
    mujoco==2.3.7
    gymnasium[box2d]
    gymnasium[mujoco]
    """.split(
        "\n"
    )
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        print("REQ:", req)
        sreqs.append(req)
    print("SREQS:", sreqs)
    return modal.Image.debian_slim(python_version="3.11.5").apt_install("swig").pip_install(sreqs)


modal_image = mk_image()

modal_app = modal.App(name="yubo")
