import torch


class KRandomizedT:
    def __init__(self, train_x: torch.Tensor):
        assert isinstance(train_x, torch.Tensor)
        assert train_x.ndim == 2
        n, d = train_x.shape
        assert n >= 1 and d >= 1
        self._train_x = train_x.detach().cpu().to(dtype=torch.float64)
        sobol = torch.quasirandom.SobolEngine(dimension=int(d), scramble=True)
        u = sobol.draw(1).reshape(int(d)).to(dtype=torch.float64)
        l_min = 1e-2
        l_max = 1.0
        self._lengthscales = torch.exp(
            torch.log(torch.tensor(l_min, dtype=torch.float64))
            + u
            * (
                torch.log(torch.tensor(l_max, dtype=torch.float64))
                - torch.log(torch.tensor(l_min, dtype=torch.float64))
            )
        )

        Xs = self._train_x / self._lengthscales
        xs2 = (Xs**2).sum(dim=1)
        D2 = xs2[:, None] + xs2[None, :] - 2.0 * (Xs @ Xs.t())
        D2 = torch.clamp(D2, min=0.0)
        K = torch.exp(-0.5 * D2)
        K = 0.5 * (K + K.t())
        K.fill_diagonal_(1.0)
        self._K = K

    def sub_k(
        self, idxs: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(idxs, torch.Tensor)
        assert idxs.ndim == 1
        assert isinstance(x, torch.Tensor)
        assert x.ndim in (1, 2)

        idxs = idxs.to(dtype=torch.int64)
        Kxx = self._K.index_select(0, idxs).index_select(1, idxs)

        Xq = x.detach().cpu().to(dtype=torch.float64)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        b, d = Xq.shape
        assert d == self._train_x.shape[1]

        Z = self._train_x.index_select(0, idxs)
        Q = Xq / self._lengthscales
        Zs = Z / self._lengthscales

        q2 = (Q**2).sum(dim=1)
        z2 = (Zs**2).sum(dim=1)
        D2 = q2[:, None] + z2[None, :] - 2.0 * (Q @ Zs.t())
        D2 = torch.clamp(D2, min=0.0)
        Kx = torch.exp(-0.5 * D2)

        if x.ndim == 1:
            Kx = Kx[0]

        return Kxx, Kx
