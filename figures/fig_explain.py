import numpy as np
import torch


def show(x):
    if hasattr(x, "detach"):
        x = x.detach().numpy()

    return " ".join([str(xx) for xx in x.flatten().tolist()])


def _mk_mesh():
    x_1, x_2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    xs = torch.tensor(list(zip(x_1.flatten(), x_2.flatten())))
    return xs, x_1, x_2


def _dump_mesh(tag, x_1, x_2, y):
    for i in range(len(y)):
        xx_1 = x_1[i, :]
        xx_2 = x_2[i, :]
        yy = y[i, :]
        print(f"{tag}: {show(yy)} {show(xx_1)} {show(xx_2)}")


def mean_contours(gp):
    xs, x_1, x_2 = _mk_mesh()
    ys = gp.posterior(xs).mean.detach().numpy()
    y = ys.reshape(x_1.shape)
    _dump_mesh("MEAN", x_1, x_2, y)


def var_contours(gp):
    xs, x_1, x_2 = _mk_mesh()
    ys = gp.posterior(xs).variance.detach().numpy()
    y = ys.reshape(x_1.shape)
    _dump_mesh("VAR", x_1, x_2, y)


def _calc_p_max_from_Y(Y):
    is_best = torch.argmax(Y, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)
    p_max = torch.zeros(Y.shape[-1])
    p_max[idcs] = counts / Y.shape[0]
    return p_max


def pmax_contours(gp):
    xs, x_1, x_2 = _mk_mesh()
    mvn = gp.posterior(xs)
    y = mvn.sample(torch.Size([1024])).squeeze()
    p_max = _calc_p_max_from_Y(y).reshape(x_1.shape)
    _dump_mesh("PMAX", x_1, x_2, p_max)


if __name__ == "__main__":
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_tag = "f:ackley-2d"
    num_arms = 4

    seed = 7
    env_conf = get_env_conf(env_tag, seed)
    policy = default_policy(env_conf)
    default_num_X_samples = max(64, 10 * num_arms)

    opt = Optimizer(env_conf, policy, num_arms)

    for i_iter in range(10):
        opt.collect_trace(ttype="mtav_ts_fast", num_iterations=1)
        acqf = opt._designers["mtav_ts_fast"].fig_last_acqf.acq_function
        x_arms = opt._designers["mtav_ts_fast"].fig_last_arms

        if i_iter in [0, 1, 10]:
            for x in x_arms:
                print(f"ITER_ARMS_{i_iter}:", show(x))

        if i_iter == 3:
            mean_contours(acqf.model)
            # pmax_contours(acqf.model)
            # var_contours(acqf.model)

            print("X_MAX:", show(acqf.X_max))
            for x in x_arms:
                print("X_ARMS:", show(x))
            for x in acqf.X_samples:
                print("X_SAMPLES:", show(x))
