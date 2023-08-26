import torch


class TS_GP:
    def __init__(self, x, F, GP, opt="general", epsilon=0.001, D=1.0, multistart=25, minimizer="L-BFGS-B", grid=100):
        """
        Create the object

        Args:
                x - initial number of points
                F - lambda for evaluation
                GP - Gaussian Process object
        """
        self.minimizer = minimizer
        self.x = x
        self.F = F
        self.y = F(x)
        self.GP = GP
        self.t = 1.0
        self.opt = opt
        self.grid = grid
        self.epsilon = epsilon
        self.multistart = multistart
        self.fit_gp(self.x, self.y)

    def fit_gp(self, x, y, iterative=False):
        self.GP.fit_gp(x, y, iterative=iterative)
        return None

    def sample_point(self, xtest=None):
        if (self.opt == "first_order") and (self.GP.admits_first_order):
            (xnext, fxnext) = self.GP.sample_and_optimize(xtest, multistart=self.multistart, minimizer=self.minimizer, grid=self.grid)
            return (xnext, fxnext)

        elif self.opt == "iterative":
            (xnext, fxnext) = self.GP.sample_iteratively_max(xtest, multistart=self.multistart, minimizer=self.minimizer, grid=self.grid)
            return (xnext, fxnext)
        else:
            if xtest is None:
                raise AssertionError("Cannnot run a general kernel (an approximation) without specified grid")
            else:
                (xnext, value) = self.GP.sample_and_max(xtest)
                return (xnext, value)

    def isin(self, xnext):
        for v in self.x:
            if torch.norm(v - xnext) < self.epsilon:
                return True

        return False

    def step(self, xtest=None):
        self.fit_gp(self.x, self.y)
        (xnext, _) = self.sample_point(xtest)
        xnext = xnext.view(-1, 1)
        reward = self.F(torch.t(xnext))

        if not self.isin(xnext[:, 0]):
            self.x = torch.cat((self.x, torch.t(xnext)), dim=0)
            self.y = torch.cat((self.y, reward), dim=0)

        self.t += 1
        return (reward, torch.t(xnext))

    """
    def eig_gap(self):
        (mu, s) = self.GP.mean_var(xtest)
        ss, _ = torch.max(s)
        return ss
    """
