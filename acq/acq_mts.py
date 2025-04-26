import numpy as np
import torch
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.sampling import draw_sobol_samples

import acq.acq_util as acq_util


class AcqMTS:
    def __init__(self, model, use_stagger=False, include_sobol=False, num_iterations=30, init_style="find"):
        self._model = model
        self._num_iterations = num_iterations
        self._init_style = init_style
        self._use_stagger = use_stagger
        self._include_sobol = include_sobol
        self._s_min = 1e-6

        self._X_0 = self._model.train_inputs[0].detach()
        self._num_dim = self._X_0.shape[-1]

        self._bounds = acq_util.default_bounds(self._num_dim).to(self._X_0)

    def draw(self, num_arms):
        if len(self._model.train_inputs[0]) == 0:
            return (
                draw_sobol_samples(
                    bounds=self._bounds,
                    n=num_arms,
                    q=1,
                )
                .squeeze(-2)
                .to(self._X_0)
            )

        mp_sampler = get_matheron_path_model(model=self._model, sample_shape=torch.Size([num_arms]))

        # X_init ~ num_arms X 1 X num_dim
        if self._init_style == "ts":
            X_init = self._thompson_sample_measurements(num_arms)
        elif self._init_style == "find":
            X_init = torch.tile(acq_util.find_max(self._model, self._bounds), dims=(num_arms, 1)).unsqueeze(1)
        elif self._init_style == "meas":
            X_init = torch.tile(self._best_measured(), dims=(num_arms, 1)).unsqueeze(1)
        else:
            assert False, f"Unknown init_style = {self._init_style}"

        Y_best = mp_sampler(X_init)

        for _ in range(self._num_iterations):
            self._iterate_(mp_sampler, X_init, Y_best)

        return X_init.squeeze(-2)

    def _best_measured(self):
        X = self._model.train_inputs[0]
        Y = self._model.train_targets
        i = np.random.choice(np.where(Y == Y.max())[0])
        return X[i, :]

    def _thompson_sample_measurements(self, num_arms):
        X = self._model.train_inputs[0].detach().numpy()
        Y = self._model.train_targets.detach().numpy()

        if len(Y) == 1:
            X_ts = X[[0], :]
        else:
            n = len(Y)
            se = Y.std() / np.sqrt(n)

            Y = np.expand_dims(Y, -1)
            Y = Y + se * np.random.normal(size=(n, num_arms))
            i = np.where(Y == Y.max(axis=0, keepdims=True))[0]
            if len(i) > num_arms:
                i = np.random.choice(i, size=(num_arms,))

            X_ts = X[i, :]

        X_ts = np.expand_dims(X_ts, 1)
        assert X_ts.shape == (num_arms, 1, X.shape[-1]), (X_ts.shape, num_arms, X.shape[-1])

        return torch.as_tensor(X_ts)

    def _iterate_(self, mp_sampler, X_best, Y_best):
        num_arms = X_best.shape[0]
        s_min = self._s_min
        s_max = 1
        u = torch.rand(size=(num_arms, 1))
        if self._use_stagger:
            l_s_min = torch.log(torch.tensor(s_min)).to(self._X_0)
            l_s_max = torch.log(torch.tensor(s_max)).to(self._X_0)
            s = torch.exp(l_s_min + (l_s_max - l_s_min) * u)
        else:
            s = s_min + (s_max - s_min) * u

        # assert s.max() <= 1, s
        assert s.min() > 0, s
        s = s.unsqueeze(-1)

        X_unif = draw_sobol_samples(
            bounds=self._bounds,
            n=num_arms,
            q=1,
        ).to(self._X_0)

        X_perturbed = X_best + s * (X_unif - X_best)
        Y_perturbed = mp_sampler(X_perturbed)

        i = Y_perturbed > Y_best
        i = i.squeeze(-1).squeeze(-1)
        X_best[i] = X_perturbed[i]
        Y_best[i] = Y_perturbed[i]
