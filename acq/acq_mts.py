import torch
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.sampling import draw_sobol_samples

import acq.acq_util as acq_util


class AcqMTS:
    def __init__(self, model, use_stagger=True):
        self._model = model
        self._num_iterations = 30
        self._use_stagger = use_stagger
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

        # X_best ~ num_arms X 1 X num_dim
        X_best = torch.tile(acq_util.find_max(self._model, self._bounds), dims=(num_arms, 1)).unsqueeze(1)
        Y_best = mp_sampler(X_best)

        for _ in range(self._num_iterations):
            self._iterate_(mp_sampler, X_best, Y_best)
        return X_best

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
