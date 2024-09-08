import numpy as np


# 1 xi ∈ [-5, 10]
# This is a guess. Waiting to hear from authors.
# https://papers.nips.cc/paper_files/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html
class TurboAckley:
    def __init__(self):
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * np.pi

    def __call__(self, x):
        # x in [-1.1]
        x = 15 * x / 2
        x = 2.5 + x
        # x in [-5, 10]
        return -self.a * np.exp(-self.b * np.sqrt((x**2).mean())) - np.exp(np.cos(self.c * x).mean()) + self.a + np.e
