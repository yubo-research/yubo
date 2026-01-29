from problems.lasso_bench_env import LassoBenchEnv


class LeukemiaEnv(LassoBenchEnv):
    def __init__(self, seed):
        super().__init__(pick_data="leukemia", seed=seed)
