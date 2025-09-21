import numpy as np
import torch

import acq.fit_gp as fit_gp
from optimizer.sobol_designer import SobolDesigner


class VecchiaDesigner:
    def __init__(self, policy, *, num_candidates_per_arm=512):
        self._policy = policy
        self._num_candidates_per_arm = num_candidates_per_arm

        self._dtype = torch.float32
        self._device = torch.empty(size=(1,)).device

        self._sobol = SobolDesigner(policy)

        self._pyvecch_ready = None

    def _ensure_pyvecch(self):
        if self._pyvecch_ready is not None:
            return self._pyvecch_ready
        try:
            from pyvecch.input_transforms import Identity  # noqa: F401
            from pyvecch.models import RFVecchia  # noqa: F401
            from pyvecch.nbrs import ExactOracle  # noqa: F401
            from pyvecch.prediction import IndependentRF  # noqa: F401
            from pyvecch.training import fit_model  # noqa: F401
        except Exception:
            self._pyvecch_ready = False
        else:
            self._pyvecch_ready = True
        return self._pyvecch_ready

    def _fit_vecchia(self, X, Y):
        from gpytorch.constraints import Interval
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.means import ZeroMean
        from pyvecch.input_transforms import Identity
        from pyvecch.models import RFVecchia
        from pyvecch.nbrs import ExactOracle
        from pyvecch.prediction import IndependentRF
        from pyvecch.training import fit_model

        assert X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]
        if len(X) == 0:
            return None

        dim = X.shape[-1]
        z = (Y - Y.mean()) / (Y.std() if Y.std() > 0 else 1.0)
        z = z.squeeze(-1)

        covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
        mean_module = ZeroMean()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

        m = max(1, int(7.2 * np.log10(max(2, len(X))) ** 2))
        neighbor_oracle = ExactOracle(X, z, m)
        prediction_strategy = IndependentRF()
        input_transform = Identity(d=dim)

        model = RFVecchia(covar_module, mean_module, likelihood, neighbor_oracle, prediction_strategy, input_transform)

        train_batch_size = int(np.minimum(len(X), 128))
        fit_model(model, train_batch_size=train_batch_size, n_window=50, maxiter=100, rel_tol=5e-3)
        model.update_transform()
        model.eval()
        model.likelihood.eval()
        return model

    def _select_candidates(self, model, X_cand, num_arms):
        assert model is not None
        with torch.no_grad():
            posterior = model.posterior(X_cand)
            mu = posterior.mean

            while mu.dim() > 1:
                mu = mu.mean(dim=0)

            mu = mu.reshape(-1)
            k = min(num_arms, mu.shape[0])
            topk = torch.topk(mu, k=k, largest=True).indices
            X_next = X_cand[topk]

        # If for any reason we have fewer than requested, pad uniquely from candidates
        if X_next.shape[-2] < num_arms:
            need = num_arms - X_next.shape[-2]
            existing = {tuple(map(float, row.tolist())) for row in X_next}
            add = []
            for row in X_cand:
                t = tuple(map(float, row.tolist()))
                if t not in existing:
                    add.append(row)
                    existing.add(t)
                    if len(add) == need:
                        break
            if len(add) > 0:
                X_next = torch.cat([X_next, torch.stack(add, dim=0)], dim=0)
        return X_next

    def __call__(self, data, num_arms):
        # Require pyvecch to be available
        if not self._ensure_pyvecch():
            raise ImportError("VecchiaDesigner requires 'pyvecch'. Please install VecchiaBO (pyvecch) from https://github.com/feji3769/VecchiaBO")

        # Cold-start: fall back to Sobol when no training data yet
        if len(data) == 0:
            return self._sobol(data, num_arms)

        # Gather training data in [0,1]^d (BoTorch space)
        Y_train, X_train = fit_gp.extract_X_Y(data, self._dtype, self._device)
        # Ensure float32 on CPU and contiguous for faiss/pyvecch
        X_train = X_train.to(dtype=torch.float32, device=torch.device("cpu")).contiguous()
        Y_train = Y_train.to(dtype=torch.float32, device=torch.device("cpu")).contiguous()

        if len(X_train) <= 1:
            return self._sobol(data, num_arms)

        model = self._fit_vecchia(X_train, Y_train)

        # Candidate set: simple Sobol cloud in TR-like box around best x
        from torch.quasirandom import SobolEngine

        if len(X_train) <= 1 or model is None:
            return self._sobol(data, num_arms)

        y_std = Y_train.std()
        if y_std.item() == 0:
            y_z = 0 * Y_train
        else:
            y_z = (Y_train - Y_train.mean()) / y_std

        x_center = X_train[torch.argmax(y_z), :].clone()

        # Lengthscale-proportional box; fall back to isotropic if unavailable
        try:
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        except Exception:
            weights = torch.ones_like(x_center)

        tr_len = 0.8
        tr_lb = torch.clamp(x_center - weights * tr_len / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * tr_len / 2.0, 0.0, 1.0)

        dim = X_train.shape[-1]
        n_candidates = max(2000, min(5000, 200 * dim))
        sobol = SobolEngine(dim, scramble=True)
        X_cand = sobol.draw(n_candidates).to(dtype=torch.float32, device=torch.device("cpu")).contiguous()
        X_cand = tr_lb + (tr_ub - tr_lb) * X_cand

        X_next = self._select_candidates(model, X_cand, num_arms)

        # Convert candidates back to policies in parameter space
        return fit_gp.mk_policies(self._policy, X_next)
