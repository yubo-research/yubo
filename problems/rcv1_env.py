from problems.lasso_bench_env import LassoBenchEnv


class Rcv1Env(LassoBenchEnv):
    def __init__(self, seed):
        super().__init__(pick_data="rcv1", seed=seed)
