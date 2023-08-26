import torch

from dpp.batched_bo import Batched_BO
from dpp.ts_gp import TS_GP


class Batched_TS_GP(TS_GP, Batched_BO):
    def __init__(self, *args, true_F=None, **kw):
        super().__init__(*args, **kw)
        if true_F is None:
            self.true_F = self.F
        else:
            self.true_F = true_F
        self.temp_pmax_samples = None
        self.start_K = None

    def sample_point(self, xtest=None):
        """
        Add option of linear programming optimization
        """
        if self.opt == "linprog":
            (xnext, fxnext) = self.GP.sample_and_optimize()
            return (xnext, fxnext)
        else:
            return super().sample_point(xtest)

    def step(self, xtest=None, batch_size=1, safe=False, hal=True, get_inference=True):
        self.check_start_K(xtest)
        # work on "hallucinated data"
        x_hal = self.x
        y_hal = self.y
        x_batch = torch.tensor([], dtype=torch.double)
        for i in range(batch_size):
            if hal:
                self.fit_gp(x_hal, y_hal)
            # use sampled reward as hallucinated reward
            (xnext, _) = self.sample_point(xtest)
            (hal_reward, _) = self.GP.mean_var(xnext.view(1, -1))
            xnext = xnext.view(-1, 1)
            if x_batch.size()[0] == 0:
                x_batch = torch.t(xnext)
            else:
                x_batch = torch.cat((x_batch, torch.t(xnext)), dim=0)
            if hal and (not safe or not self.isin_hal(xnext[:, 0], x_hal)):
                x_hal = torch.cat((x_hal, torch.t(xnext)), dim=0)
                y_hal = torch.cat((y_hal, hal_reward), dim=0)

        if hal:
            self.fit_gp(self.x, self.y)

        res = self.step_update(xtest, x_batch, safe, get_inference)
        self.t += 1
        return res
