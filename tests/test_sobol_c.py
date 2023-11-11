def test_sobol_center():
    import numpy as np

    from common.seed_all import seed_all
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    prev_data = None
    print()
    num_checks = 0
    for num_arms in [3, 4, 5]:
        seed_all(17)
        env_conf = get_env_conf("f:sphere-2d", seed=17)
        policy = default_policy(env_conf)
        opt = Optimizer(env_conf, policy, num_arms=num_arms)
        opt.collect_trace(ttype="sobol_c", num_iterations=1)
        if prev_data is not None:
            for d_p, d in zip(prev_data, opt._data):
                num_checks += 1
                assert np.all(d_p.policy.get_params() == d.policy.get_params())
        prev_data = opt._data

    assert num_checks == 7
