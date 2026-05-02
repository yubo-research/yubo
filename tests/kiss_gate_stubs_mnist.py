import torch
from torch.utils.data import Dataset


class _TinyMNIST(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return torch.zeros((1, 28, 28), dtype=torch.float32), torch.tensor(idx % 10, dtype=torch.long)
