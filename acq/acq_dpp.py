import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine


class _GP:
    def __init__(self, model) -> None:
        self.model = model
        self.d = self.model.train_inputs[0].size(1)
        self.model.eval()

    @property
    def s(self):
        return self.model.likelihood.noise.sqrt()

    def mean_var(self, x: torch.Tensor, full: bool = True):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(x)
        mean = posterior.mean
        var = posterior.covariance_matrix
        if not full:
            var = var.diag()
        return mean, var

    def sample_from_pmax(self, xtest: torch.Tensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(xtest)
        samples = posterior.sample()
        max_ind = torch.argmax(samples)
        return xtest[max_ind]


class AcqDPP:
    """
    Elvis Nava, Mojmir Mutny, and Andreas Krause. Diversified sampling for batched bayesian
        optimization with determinantal point processes. In Gustau Camps-Valls, Francisco J. R.
        Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on
        Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Re-
        search, pages 7031–7054. PMLR, 28–30 Mar 2022. URL https://proceedings.mlr.press/v151/nava22a.html.
    """

    def __init__(self, model, num_X_samples, num_runs=50, DPP_lambda=1.0, cutoff_iter=None, lambda_mode="mult"):
        """
        num_runs are the runs of the MCMC algorithm
        DPP_lambda: either a number, or a callable to get lambda a function of t (iter number)
        cutoff_iter: number of iterations after which we no longer DPP sample, just exploit with TS
        lambda_mode: 'mult' if (I + lambda sigma^-2 K), 'pow' if (I + sigma^-2 K)^lambda
        """

        self._num_runs = num_runs
        self._DPP_lambda = DPP_lambda
        self._cutoff_iter = cutoff_iter
        self._lambda_mode = lambda_mode

        X_0 = model.train_inputs[0].detach()
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype

        sobol_engine = SobolEngine(self._num_dim, scramble=True)
        self._X_samples = sobol_engine.draw(num_X_samples, dtype=self._dtype)
        # self.x = X_0
        # self.y = model.train_targets
        self.GP = _GP(model)
        self.model = model
        self.num_rounds = 1.0
        self.start_K = None

    def _check_start_K(self, xtest, fake_xtest_size=512):
        if self.start_K is None:
            # If we are in the continuous setting, then xtest=None, but we may need an xtest for some metrics
            if xtest is None:
                bounds = np.array(self.GP.bounds)
                self.fake_xtest = torch.tensor(
                    np.random.uniform(
                        low=np.tile(bounds[:, 0], (fake_xtest_size, 1)),
                        high=np.tile(bounds[:, 1], (fake_xtest_size, 1)),
                        size=(fake_xtest_size, self.GP.d),
                    ),
                    dtype=torch.double,
                )
            (_, self.start_K) = self.GP.mean_var(xtest if xtest is not None else self.fake_xtest, full=True)

    def draw(self, num_arms, first_ts=False):
        num_runs = self._num_runs

        self._check_start_K(self._X_samples)
        # get lambda value for callable
        if callable(self._DPP_lambda):
            self._DPP_lambda = self._DPP_lambda(self.num_rounds)

        # Sample first X_batch
        X_batch = torch.tensor([], dtype=torch.double)
        for i in range(num_arms):
            X_next = self.GP.sample_from_pmax(self._X_samples)
            X_next = X_next.view(-1, 1)
            if X_batch.size()[0] == 0:
                X_batch = torch.t(X_next)
            else:
                X_batch = torch.cat((X_batch, torch.t(X_next)), dim=0)
        (_, post_K_S) = self.GP.mean_var(X_batch, full=True)
        if self._lambda_mode == "mult":
            K_S = torch.eye(num_arms, dtype=torch.float64) + self._DPP_lambda * (self.GP.s**-2) * post_K_S
            det_K_S = torch.det(K_S)
        elif self._lambda_mode == "pow":
            K_S = torch.eye(num_arms, dtype=torch.float64) + (self.GP.s**-2) * post_K_S
            det_K_S = torch.det(K_S) ** self._DPP_lambda
        else:
            raise ValueError(f"Unsupported lambda_mode {self._lambda_mode}")
        log_lik = torch.log(det_K_S)
        log_lik_hist = log_lik.view(1, 1)

        # MCMC
        if self._cutoff_iter is None or self.num_rounds < self._cutoff_iter:
            while num_runs > 0:
                switch_i = np.random.randint(0 if not first_ts else 1, num_arms)  # If first was sampled with regular TS, cannot swap it during MCMC
                X_next = self.GP.sample_from_pmax(self._X_samples)
                X_batch_prop = X_batch.clone()
                X_batch_prop[switch_i] = torch.t(X_next.view(-1, 1))
                (_, post_K_S) = self.GP.mean_var(X_batch_prop, full=True)
                if self._lambda_mode == "mult":
                    K_S = torch.eye(num_arms, dtype=torch.float64) + self._DPP_lambda * (self.GP.s**-2) * post_K_S
                    det_K_S_prop = torch.det(K_S)
                elif self._lambda_mode == "pow":
                    K_S = torch.eye(num_arms, dtype=torch.float64) + (self.GP.s**-2) * post_K_S
                    det_K_S_prop = torch.det(K_S) ** self._DPP_lambda
                else:
                    raise ValueError(f"Unsupported lambda_mode {self._lambda_mode}")
                alpha = torch.min(torch.tensor(1, dtype=torch.double), det_K_S_prop / det_K_S)
                if torch.rand(1) < alpha:
                    X_batch = X_batch_prop
                    det_K_S = det_K_S_prop
                log_lik = torch.log(det_K_S)
                log_lik_hist = torch.cat((log_lik_hist, log_lik.view(1, 1)), dim=0)
                num_runs -= 1
        return X_batch
