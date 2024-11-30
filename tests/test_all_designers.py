# def test_designers_0():
#     _test_some_designers(0, 4)


# def test_designers_1():
#     _test_some_designers(1, 4)


# def test_designers_2():
#     _test_some_designers(2, 4)


# def test_designers_3():
#     _test_some_designers(3, 4)


# def _test_some_designers(k, m):
#     for i, dn in enumerate(_test_designer("random")):
#         if i % m == k:
#             print("DESIGNER:", dn)
#             _test_designer(dn)


# def _test_designer(designer):
#     from common.collector import Collector
#     from common.seed_all import seed_all
#     from optimizer.arm_best_obs import ArmBestObs
#     from optimizer.optimizer import Optimizer
#     from problems.env_conf import default_policy, get_env_conf

#     seed_all(17)
#     env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
#     policy = default_policy(env_conf)
#     if designer in ["ax", "maximin", "maximin-toroidal", "variance", "sobol_gibbon", "turbo"]:
#         num_arms = 1
#     else:
#         num_arms = 3
#     collector = Collector()
#     opt = Optimizer(
#         collector,
#         env_conf=env_conf,
#         policy=policy,
#         num_arms=num_arms,
#         arm_selector=ArmBestObs(),
#     )
#     opt.collect_trace(designer_name=designer, num_iterations=4)

#     return opt.all_designer_names()
