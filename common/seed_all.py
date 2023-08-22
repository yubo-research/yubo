import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(999999))
