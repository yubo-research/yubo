import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    torch_seed = np.random.randint(999999)
    torch.manual_seed(torch_seed)
    print("SEEDS:", seed, torch_seed)
