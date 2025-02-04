from torch.quasirandom import SobolEngine


class AcqSobol:
    def __init__(self, model):
        self.model = model

        X_0 = self.model.train_inputs[0].detach()
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype
        self._device = X_0.device

    def draw(self, num_arms):
        sobol = SobolEngine(self._num_dim, scramble=True)
        return sobol.draw(num_arms).to(dtype=self._dtype, device=self._device)
