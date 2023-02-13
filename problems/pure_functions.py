import numpy as np
from gymnasium.spaces import Box
# import problems.sphere as bf
from problems.benchmark_functions import Ackley,Beale,Branin,Bukin,CrossInTray,DropWave,DixonPrice,EggHolder,Griewank,GrLee12,Hartmann,HolderTable,Levy, Michalewicz,Powell,Rastrigin,Rosenbrock,Shubert,Shekel,SixHumpCamel,StybTang,ThreeHumpCamel


def make(name, seed):
    # TODO: noise
    _, name = name.split(":")
    name, num_dim = name.split("-")
    assert num_dim[-1] == "d"
    num_dim = int(num_dim[:-1])
    #1
    if name == "ackley":
        return PureFunctionEnv(Ackley(seed, num_dim), num_dim)
    #2
    if name == "beale":
        return PureFunctionEnv(Beale(seed, num_dim), num_dim)
    # 3
    if name == "branin":
        return PureFunctionEnv(Branin(seed, num_dim), num_dim)
    # 4
    if name == "bukin":
        return PureFunctionEnv(Bukin(seed, num_dim), num_dim)
    # 5
    if name == "crossintray":
        return PureFunctionEnv(CrossInTray(seed, num_dim), num_dim)
    # 7
    if name == "dropwave":
        return PureFunctionEnv(DropWave(seed, num_dim), num_dim)
    # 7
    if name == "dixonprice":
        return PureFunctionEnv(DixonPrice(seed, num_dim), num_dim)
    # 8
    if name == "eggHolder":
        return PureFunctionEnv(EggHolder(seed, num_dim), num_dim)
    # 9
    if name == "griewank":
        return PureFunctionEnv(Griewank(seed, num_dim), num_dim)
    # 10
    if name == "grlee12 ":
        return PureFunctionEnv(GrLee12(seed, num_dim), num_dim)
    # 11
    if name == "hartmann":
        return PureFunctionEnv(Hartmann(seed, num_dim), num_dim)
    # 12
    if name == "holdertable":
        return PureFunctionEnv(HolderTable(seed, num_dim), num_dim)
    # 13
    if name == "levy":
        return PureFunctionEnv(Levy(seed, num_dim), num_dim)
    # 14
    if name == "michalewicz":
        return PureFunctionEnv(Michalewicz(seed, num_dim), num_dim)
    # 15
    if name == "powell":
        return PureFunctionEnv(Powell(seed, num_dim), num_dim)
    # 16
    if name == "rastrigin":
        return PureFunctionEnv(Rastrigin(seed, num_dim), num_dim)
    # 17
    if name == "rosenbrock":
        return PureFunctionEnv(Rosenbrock(seed, num_dim), num_dim)
    # 18
    if name == "shubert":
        return PureFunctionEnv(Shubert(seed, num_dim), num_dim)
    # 19
    if name == "shekel":
        return PureFunctionEnv(Shekel(seed, num_dim), num_dim)
    # 20
    if name == "sixhumpcamel":
        return PureFunctionEnv(SixHumpCamel(seed, num_dim), num_dim)
    # 21
    if name == "stybtang":
        return PureFunctionEnv(StybTang(seed, num_dim), num_dim)
    # 22
    if name == "threehumpcamel":
        return PureFunctionEnv(ThreeHumpCamel(seed, num_dim), num_dim)

    assert False, name


# all function maxes are 0
# all domains are [-1,1]**D


class PureFunctionEnv:
    def __init__(self, function, num_dim):
        self._function = function

        self.observation_space = Box(low=0.0, high=1.0, dtype=np.float32)

        self.action_space = Box(low=-np.ones(num_dim), high=np.ones(num_dim), dtype=np.float32)

    def step(self, action):
        # state, reward, done = env.step(action)[:3]
        return 1, self._function(action), True, None

    def reset(self, seed):
        return 0, None

    def close(self):
        pass
