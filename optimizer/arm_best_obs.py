import numpy as np


class ArmBestObs:
    def __init__(self):
        pass

    def __call__(self, data):
        rets = np.array([d.trajectory.rreturn for d in data])
        try:
            i = np.random.choice(np.where(rets == rets.max())[0])
        except ValueError:
            breakpoint()

        return data[i].policy, data[i].trajectory.rreturn
