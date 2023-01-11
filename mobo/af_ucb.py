import torch

class AFUCB:
    def __init__(self, llambda):
        self._llambda = llambda

    def __call__(self, mu, var):
        # set_trace()
        assert var.item() >= 0, (mu, var)
        return mu + self._llambda * torch.sqrt(var)
