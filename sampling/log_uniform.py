import numpy as np
import torch


def torch_log_uniform(s_min, s_max):
    u = torch.rand(size=(1,))
    l_s_min = torch.log(torch.tensor(s_min))
    l_s_max = torch.log(torch.tensor(s_max))
    return torch.exp(l_s_min + (l_s_max - l_s_min) * u)[0]


def np_log_uniform(s_min, s_max):
    u = np.random.uniform(size=(1,))
    l_s_min = np.log(s_min)
    l_s_max = np.log(s_max)
    return np.exp(l_s_min + (l_s_max - l_s_min) * u)[0]
