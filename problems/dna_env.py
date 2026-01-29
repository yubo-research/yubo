from problems.lasso_bench_env import LassoBenchEnv


class DnaEnv(LassoBenchEnv):
    def __init__(self, seed):
        super().__init__(pick_data="dna", seed=seed)
