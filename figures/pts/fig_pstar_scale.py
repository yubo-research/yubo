import time

import modal
import numpy as np

from acq.acq_util import calc_p_max
from common.collector import Collector
from experiments.dist_modal import DistModal, collect
from experiments.modal_image import mk_image
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf

modal_image = mk_image()


# _mode = "all"
_mode = "sphere3"

_num_dims = [5]

_app_name = "yubo-fig-pstar-scale"
app = modal.App(name=_app_name)


@app.function(image=modal_image, max_containers=100, timeout=3 * 60 * 60)  # , gpu="H100")
def calc_pstar_scales(d_args):
    env_tag, designer_name, num_arms, num_samples, num_dim = (
        d_args["env_tag"],
        d_args["designer_name"],
        d_args["num_arms"],
        d_args["num_samples"],
        d_args["num_dim"],
    )

    data = []

    seed = np.random.randint(999999)
    env_conf = get_env_conf(env_tag, problem_seed=seed, noise_seed_0=seed + 1)
    policy = default_policy(env_conf)

    collector_log = Collector()
    opt = Optimizer(
        collector_log,
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise_measurement=None,
    )

    t_0 = time.time()
    for i_iter in range(max(30, 2 * num_dim)):
        trace = opt.collect_trace(designer_name=designer_name, num_iterations=1)

        if opt.last_designer.fig_last_acqf == "sobol":
            assert i_iter == 0, i_iter
            data.append(None)
            continue
        acqf = opt.last_designer.fig_last_acqf.acq_function
        # X_samples = acqf.draw(num_samples)

        opt.last_designer(opt._data, num_arms=num_samples)
        X_samples = opt.last_designer.fig_last_arms

        p_max = calc_p_max(acqf.model, X_samples, num_Y_samples=1024)
        X_samples = X_samples.numpy()

        if _mode == "sphere3":
            bias = (X_samples - 0.65).mean(axis=0)
            rmse = np.sqrt(((X_samples - 0.65) ** 2).mean(axis=0))
        elif _mode == "all":
            bias = np.nan
        else:
            assert False, _mode

        num_dim = X_samples.shape[-1]
        scale = np.prod(X_samples.std(axis=0)) ** (1 / num_dim)

        data.append((bias, scale, rmse, p_max, time.time() - t_0, trace[-1].rreturn))

    return designer_name, num_dim, env_tag, data[1:]


def dist_pstar_scales_all_funcs(designer, num_dim):
    from experiments.func_names import funcs_nd

    if _mode == "sphere3":
        funcs = ["sphere3"] * 10
        prefix = "g"
    elif _mode == "all":
        funcs = funcs_nd
        prefix = "f"
    else:
        assert False, _mode

    all_d_args = []
    for func in funcs:
        env_tag = f"{prefix}:{func}-{num_dim}d"
        d = dict(
            env_tag=env_tag,
            designer_name=designer,
            num_arms=1,
            num_samples=64,
            num_dim=num_dim,
        )
        print("DIST:", d)
        all_d_args.append(d)

    return all_d_args


def distribute(designer, job_fn, dry_run):
    all_d_args = []
    for num_dim in _num_dims:
        all_d_args.extend(dist_pstar_scales_all_funcs(designer, num_dim=num_dim))

    if not dry_run:
        dm = DistModal(_app_name, "calc_pstar_scales", job_fn)
        dm(all_d_args)


def collect_all(job_fn):
    import pickle

    data = {}

    def _cb(result):
        designer_name, num_dim, env_tag, datum = result
        if designer_name not in data:
            data[designer_name] = {}
        d = data[designer_name]
        if num_dim not in d:
            d[num_dim] = []
        d[num_dim].append(datum)

    collect(job_fn, _cb)

    for designer_name in data:
        fn = f"fig_data/sts/pstar_{_mode}_{designer_name}.pkl"
        with open(fn, "wb") as f:
            pickle.dump(data[designer_name], f)
        print("WROTE:", fn)


@app.local_entrypoint()
def spawn_all(cmd: str, job_fn: str, dry_run: bool = False, designer: str = None):
    if cmd == "dist":
        distribute(designer, job_fn, dry_run)
    elif cmd == "collect":
        collect_all(job_fn)
    else:
        assert False, "Bad command"
