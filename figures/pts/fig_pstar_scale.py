import numpy as np

from common.collector import Collector
from optimizer.arm_best_obs import ArmBestObs
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf
from sampling.stagger_sampler import StaggerSampler


def collect_pstar_scales(env_tag, designer_name, num_arms, num_samples):
    data = []

    env_conf = get_env_conf(env_tag, problem_seed=17, noise_seed_0=1)
    policy = default_policy(env_conf)

    arm_selector = ArmBestObs()

    collector_log = Collector()
    opt = Optimizer(
        collector_log,
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise=None,
        num_obs=1,
        arm_selector=arm_selector,
    )

    for _ in range(31):
        opt.collect_trace(designer_name=designer_name, num_iterations=1)
        acqf = opt.last_designer.fig_last_acqf.acq_function
        # x_arms = opt.last_designer.fig_last_arms

        if not isinstance(acqf.X_samples, str):
            X_samples = acqf.X_samples
        else:
            pss = StaggerSampler(acqf.model, acqf.X_max, num_samples=num_samples)
            pss.refine(acqf.k_mcmc)
            X_samples = pss.samples()

        num_dim = X_samples.shape[-1]
        scale = np.prod(X_samples.std(dim=0).numpy()) ** (1 / num_dim)
        data.append(scale)

    return np.array(data)[1:]


def collect_pstar_scales_all_funcs(num_dim):
    from experiments.func_names import funcs_nd

    data = []
    for func in funcs_nd:
        env_tag = f"f:{func}-{num_dim}d"
        data.append(
            collect_pstar_scales(
                env_tag,
                designer_name="mtv-pss-ts",
                num_arms=1,
                num_samples=64,
            )
        )
    return np.array(data)


def collect_all():
    import pickle

    data = {}
    for num_dim in [3, 10, 30, 100]:
        data[num_dim] = collect_pstar_scales_all_funcs(num_dim=num_dim)

    with open("fig_data/pts/pstar_scale.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    pass
