import numpy as np
from gymnasium.spaces import Box

# import problems.sphere as bf
from problems.benchmark_functions import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    CrossInTray,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    GrLee12,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    Shubert,
    SixHumpCamel,
    Sphere,
    StybTang,
    ThreeHumpCamel,
)


def make(name, seed):
    # TODO: noise
    _, name = name.split(":")
    name, num_dim = name.split("-")
    assert num_dim[-1] == "d"
    num_dim = int(num_dim[:-1])
    if name == "sphere" or name == "Sphere":
        return PureFunctionEnv(Sphere(seed, num_dim), num_dim)
    # 1
    if name == "ackley" or name == "Ackley":
        return PureFunctionEnv(Ackley(seed, num_dim), num_dim)
    # 2
    if name == "beale" or name == "Beale":
        return PureFunctionEnv(Beale(seed, num_dim), num_dim)
    # 3
    if name == "branin" or name == "Branin":
        return PureFunctionEnv(Branin(seed, num_dim), num_dim)
    # 4
    if name == "bukin" or name == "Bukin":
        return PureFunctionEnv(Bukin(seed, num_dim), num_dim)
    # 5
    if name == "crossintray" or name == "CrossInTray":
        return PureFunctionEnv(CrossInTray(seed, num_dim), num_dim)
    # 7
    if name == "dropwave" or name == "DropWave":
        return PureFunctionEnv(DropWave(seed, num_dim), num_dim)
    # 7
    if name == "dixonprice" or name == "DixonPrice":
        return PureFunctionEnv(DixonPrice(seed, num_dim), num_dim)
    # 8
    if name == "eggHolder" or name == "EggHolder":
        return PureFunctionEnv(EggHolder(seed, num_dim), num_dim)
    # 9
    if name == "griewank" or name == "GrieWank":
        return PureFunctionEnv(Griewank(seed, num_dim), num_dim)
    # 10
    if name == "grlee12" or name == "GrLee12":
        return PureFunctionEnv(GrLee12(seed, num_dim), num_dim)
    # 11
    if name == "hartmann" or name == "Hartmann":
        return PureFunctionEnv(Hartmann(seed, num_dim), num_dim)
    # 12
    if name == "holdertable" or name == "HolderTable":
        return PureFunctionEnv(HolderTable(seed, num_dim), num_dim)
    # 13
    if name == "levy" or name == "Levy":
        return PureFunctionEnv(Levy(seed, num_dim), num_dim)
    # 14
    if name == "michalewicz" or name == "Michalewicz":
        return PureFunctionEnv(Michalewicz(seed, num_dim), num_dim)
    # 15
    if name == "powell" or name == "Powell":
        return PureFunctionEnv(Powell(seed, num_dim), num_dim)
    # 16
    if name == "rastrigin" or name == "Rastrigin":
        return PureFunctionEnv(Rastrigin(seed, num_dim), num_dim)
    # 17
    if name == "rosenbrock" or name == "Rosenbrock":
        return PureFunctionEnv(Rosenbrock(seed, num_dim), num_dim)
    # 18
    if name == "shubert" or name == "Shubert":
        return PureFunctionEnv(Shubert(seed, num_dim), num_dim)
    # 19
    if name == "shekel" or name == "Shekel":
        return PureFunctionEnv(Shekel(seed, num_dim), num_dim)
    # 20
    if name == "sixhumpcamel" or name == "SixHumpCamel":
        return PureFunctionEnv(SixHumpCamel(seed, num_dim), num_dim)
    # 21
    if name == "stybtang" or name == "StybTang":
        return PureFunctionEnv(StybTang(seed, num_dim), num_dim)
    # 22
    if name == "threehumpcamel" or name == "ThreeHumpCamel":
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
