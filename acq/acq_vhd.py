import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.scale_free_sampler import scale_free_sampler

from .acq_util import torch_random_choice


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_samples=256):
        self._X_train = X_train
        self._Y_train = Y_train
        self._num_refinements = 100
        self._num_samples = num_samples
        self._b_raasp = False

        if k > 0 and len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=k)
        else:
            self._enn = None

    def _get_max(self):
        assert len(self._X_train) > 0
        # TODO: ts-max when YVar
        # TODO: Maybe use ENN
        Y = self._Y_train
        i = torch_random_choice(torch.where(Y == Y.max())[0])
        return self._X_train[i, :]

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            return torch.rand(size=(num_arms, self._X_train.shape[-1]))

        # TODO: MTV for batches

        X_a = []
        for _ in range(num_arms):
            X_0 = self._get_max()
            # TODO: Make scale_free_sampler produce a batch of num_samples samples
            X_cand = torch.stack([scale_free_sampler(X_0, b_raasp=self._b_raasp) for _ in range(self._num_samples)]).squeeze(1)
            if self._enn is not None:
                Y_cand = self._enn(X_cand).sample()
                i = torch_random_choice(torch.where(Y_cand == Y_cand.max())[0])
            else:
                i = torch.randint(low=0, high=len(X_cand), size=(1,))
            X_a.append(X_cand[i])

        return torch.stack(X_a)

    def _xxx_draw_1(self):
        X = self._X_max
        for _ in range(self._num_refinements):
            X_a = scale_free_sampler(X, b_raasp=self._b_raasp)
            if self._enn is not None:
                # TODO: enn.joint_2(X, X_a)
                Y = self._enn(X).sample()
                Y_a = self._enn(X_a).sample()
                if Y_a > Y:
                    X = X_a
            else:
                X = X_a
        return X
