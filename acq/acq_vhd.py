import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.lhd import latin_hypercube_design
from sampling.scale_free_sampler import scale_free_sampler

from .acq_util import torch_random_choice


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1):
        self._X_train = X_train
        self._Y_train = Y_train
        self._num_refinements = 100
        self._b_raasp = False

        if k > 0 and len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=k)
        else:
            self._enn = None

        self._X_max = self._get_max()

    def _get_max(self):
        if len(self._X_train) == 0:
            return torch.rand(size=(1, self._X_train.shape[-1]))
        # TODO: ts-max when YVar
        # TODO: Maybe use ENN
        Y = self._Y_train
        i = torch_random_choice(torch.where(Y == Y.max())[0])
        return self._X_train[i, :]

    def draw(self, num_arms):
        # TODO: MTV for batches
        return torch.stack([self._draw_1() for _ in range(num_arms)])

    def _draw_1(self):
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
