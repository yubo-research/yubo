import numpy as np
import torch

from problems.env_conf import default_policy, get_env_conf
from problems.pure_functions import PureFunctionEnv

PureFunctionEnv.ALPHA = 1.0


def expository_problem():
    env_tag = "f:ackley-2d"

    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_conf = get_env_conf(env_tag, seed)
    policy = default_policy(env_conf)
    return env_conf, policy


def show(x):
    if hasattr(x, "detach"):
        x = x.detach().numpy()

    return " ".join([str(xx) for xx in x.flatten().tolist()])


def mk_mesh():
    x_1, x_2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    xs = torch.tensor(list(zip(x_1.flatten(), x_2.flatten())))
    return xs, x_1, x_2


def dump_mesh(out_dir, tag, x_1, x_2, y):
    with open(f"{out_dir}/{tag}", "w") as f:
        for i in range(len(y)):
            xx_1 = x_1[i, :]
            xx_2 = x_2[i, :]
            yy = y[i, :]
            f.write(f"{show(yy)} {show(xx_1)} {show(xx_2)}\n")


def mean_func_contours(out_dir, func):
    xs, x_1, x_2 = mk_mesh()
    assert False, "NYI"
    # ys = gp.posterior(xs).mean.detach().numpy()
    # y = ys.reshape(x_1.shape)
    # dump_mesh(out_dir, "mean", x_1, x_2, y)


def mean_gp_contours(out_dir, gp):
    xs, x_1, x_2 = mk_mesh()
    ys = gp.posterior(xs).mean.detach().numpy()
    y = ys.reshape(x_1.shape)
    dump_mesh(out_dir, "mean", x_1, x_2, y)


def var_contours(out_dir, gp):
    xs, x_1, x_2 = mk_mesh()
    ys = gp.posterior(xs).variance.detach().numpy()
    y = ys.reshape(x_1.shape)
    dump_mesh(out_dir, "var", x_1, x_2, y)


def _calc_p_max_from_Y(Y):
    is_best = torch.argmax(Y, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)
    p_max = torch.zeros(Y.shape[-1])
    p_max[idcs] = counts / Y.shape[0]
    return p_max


def pmax_contours(out_dir, gp):
    xs, x_1, x_2 = mk_mesh()
    mvn = gp.posterior(xs)
    y = mvn.sample(torch.Size([1024])).squeeze()
    p_max = _calc_p_max_from_Y(y).reshape(x_1.shape)
    dump_mesh(out_dir, "pmax", x_1, x_2, p_max)
