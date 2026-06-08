from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch


class BinDataLoader:
    """High-performance dataloader for pre-tokenized .bin files using memmap."""

    def __init__(self, filename: str, b: int, t: int) -> None:
        self.b = b
        self.t = t

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Binary data file not found: {filename}")

        # memmap the tokens (assuming uint16 for standard GPT tokenizers)
        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        self.pos = 0

    def get_batch(self, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples a random batch from the data."""
        rng = np.random.default_rng(seed)

        # Pick B random offsets, ensuring we have enough room for T tokens + 1 target
        ix = rng.integers(0, self.num_tokens - self.t - 1, size=(self.b,))

        x = torch.stack([torch.from_numpy((self.data[i : i + self.t]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.t]).astype(np.int64)) for i in ix])

        return x.to(device), y.to(device)

    def iterator(self, seed: int, device: str) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Infinite iterator of random batches."""
        local_seed = seed
        while True:
            yield self.get_batch(local_seed, device)
            local_seed += 1
