import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, nearest_neighbor, random_directions
from sampling.lhd import latin_hypercube_design


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_candidates_per_arm=100):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_candidates_per_arm = num_candidates_per_arm
        self._num_dim = self._X_train.shape[-1]
        self._seed = np.random.randint(0, 9999)

        if len(self._X_train) > 0:
            self._enn_1 = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=1)
            if k > 0:
                self._enn_ts = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=k)
            else:
                self._enn_ts = None
        else:
            self._enn_1 = None
            self._enn_ts = None

    def get_max(self):
        assert len(self._X_train) > 0
        if False:  # TODO: study noisy observations self._enn_ts:
            Y = self._enn_ts(self._X_train).sample()
        else:
            Y = self._Y_train

        i = np.random.choice(np.where(Y == Y.max())[0])
        return self._X_train[[i], :]

    def _thompson_sample(self, x, num_arms):
        num_dim = x.shape[-1]
        num_candidates_per_arm = len(x) // num_arms
        assert len(x) / num_arms == num_candidates_per_arm
        y = self._enn_ts.posterior(x).sample()

        y = np.reshape(y, newshape=(num_arms, num_candidates_per_arm))
        x = np.reshape(x, newshape=(num_arms, num_candidates_per_arm, num_dim))
        i = np.where(y == y.max(axis=1, keepdims=True))
        return x[i]

    def draw(self, num_arms):
        num_explore = self._num_explore(num_arms)
        x_a = np.empty(shape=(0, self._num_dim))

        if num_explore > 0 or len(self._X_train) == 0:
            if len(self._X_train) == 0:
                x_a = np.append(x_a, 0.5 + np.zeros(shape=(1, self._num_dim)), axis=0)
                num_explore = num_arms

            num_left = max(0, num_explore - len(x_a))
            print("EXPLORE:", num_left, len(x_a))
            if num_left == 0:
                assert len(x_a) == num_arms, (len(x_a), num_arms)
                return x_a
            else:
                # TODO: random, sobol
                if len(self._X_train) == 0:
                    xs = np.random.uniform(size=(num_left, self._num_dim))
                    xs = latin_hypercube_design(num_left, self._num_dim, seed=self._seed + len(self._X_train) + 1)
                else:
                    # xs = np.random.uniform(size=(num_left, self._num_dim))
                    # xs = self._thompson_sample(np.random.uniform(size=(100 * num_left, self._num_dim)), num_left, 100)
                    # xs = self._find_isolated(x_a, num_left)
                    xs = latin_hypercube_design(num_left, self._num_dim, seed=self._seed + len(self._X_train) + 1)
                    pass

                x_a = np.append(x_a, xs, axis=0)

        assert len(x_a) <= num_arms, (len(x_a), num_arms)
        num_left = num_arms - len(x_a)
        if num_left == 0:
            return x_a

        # TODO: study noisy observations;  move X_0 inside loop
        print("VHD:", num_left)
        num_candidates = num_left  # self._num_candidates_per_arm * num_left
        x_0 = np.tile(self.get_max(), reps=(num_candidates, 1))
        assert x_0.shape == (num_candidates, self._num_dim), x_0.shape

        u = random_directions(num_left, self._num_dim)
        x_cand = farthest_neighbor(self._enn_1, x_0, u)

        # x_cand = find_farthest_neighbor(self._enn, x_0)
        # x_cand = (2 / 3.0) * x_cand + (1 / 3.0) * x_0

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())
        if num_candidates > num_left:
            x_a = np.append(x_a, self._thompson_sample(x_cand, num_left, self._num_candidates_per_arm), axis=0)
        else:
            x_a = np.append(x_a, x_cand, axis=0)
        assert len(x_a) == num_arms, (len(x_a), num_arms)
        return x_a


def find_farthest_neighbor(enn: EpsitemicNearestNeighbors, x_0: np.ndarray):
    import torch

    idx_0, dist_0 = enn.about_neighbors(x_0, k=1)
    assert dist_0 == 0, (idx_0, dist_0)

    X_0 = torch.tensor(x_0)
    X = X_0.clone() + 1e-6 * torch.randn(size=X_0.shape)
    X_0.requires_grad = False
    X.requires_grad = True

    optimizer = torch.optim.Adam([X], lr=0.03)

    for _ in range(1000):
        x_last = X.clone().detach().numpy()
        optimizer.zero_grad()
        neg_dist_2 = -((X - X_0) ** 2).sum(dim=1)
        neg_dist_2.backward()
        optimizer.step()

        x = X.detach().numpy()
        idx = nearest_neighbor(enn, x)[0]
        # print(torch.sqrt(-neg_dist_2).item(), idx, idx_0)
        if idx != idx_0:
            break
    else:
        assert False, "Search failed"

    # TODO: Bisection search to find boundary, which is
    #  between x_last and x.
    return x_last
