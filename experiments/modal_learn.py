import modal

app = modal.App(name="botorch_example")


def mk_image():
    reqs = """
    numpy==2.0.0
    scipy==1.13.1
    torch==2.1.0
    botorch==0.9.2
    gpytorch==1.11
    gymnasium==0.29.1
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
    return modal.Image.debian_slim(python_version="3.11.5").pip_install(sreqs)


image = mk_image()


@app.function(image=image)
def my_function(**kwargs):
    print(kwargs)
    import torch

    num_dim = int(kwargs["num_dim"])

    print("NAME:", __name__, num_dim)

    X = torch.rand(size=(10, num_dim))
    Y = ((X - 0.03) ** 2).sum(dim=1) + torch.randn(size=(10,))
    print(X)
    print(Y)


if __name__ == "__main__":
    my_function.local(num_dim=3)
