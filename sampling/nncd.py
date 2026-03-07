def nncd_weights(y, x, iters_per_dimension=3, eps=1e-6):
    B, N, D = x.shape
    import numpy as np

    assert x.ndim == 3 and x.shape == (B, N, D)
    assert y.ndim == 3 and y.shape[0] == B and y.shape[1] == N and y.shape[2] == 1
    y = y.astype(np.float64)
    x = x.astype(np.float64)
    w = np.full((B, D), 1.0 / D, dtype=np.float64)
    y2 = np.squeeze(y, axis=-1)
    G = np.einsum("bnd,bnk->bdk", x, x)
    c = np.einsum("bnd,bn->bd", x, y2)

    def _proj_simplex(W):
        B_, D_ = W.shape
        U = -np.sort(-W, axis=1)
        cssv = np.cumsum(U, axis=1) - 1.0
        ind = np.arange(1, D_ + 1)
        cond = U - cssv / ind > 0
        rho = cond.sum(axis=1) - 1
        theta = cssv[np.arange(B_), rho] / (rho + 1)
        return np.maximum(W - theta[:, None], 0.0)

    def _f_obj_var(W):
        GW = np.einsum("bdm,bm->bd", G, W)
        return 0.5 * np.einsum("bd,bd->b", W, GW) - np.einsum("bd,bd->b", W, c)

    prev = _f_obj_var(w)
    for _ in range(max(1, iters_per_dimension)):
        for i in range(D):
            Gi = G[:, i, :]
            denom = G[:, i, i]
            assert np.all(denom > 1e-12)
            Gw = np.einsum("bd,bd->b", Gi, w)
            numer = c[:, i] - Gw + G[:, i, i] * w[:, i]
            w[:, i] = numer / denom
        w = _proj_simplex(w)
        if eps is not None and eps > 0:
            cur = _f_obj_var(w)
            denomf = np.maximum(1.0, np.abs(prev))
            rel = np.max(np.abs(cur - prev) / denomf)
            prev = cur
            if rel < eps:
                break
    return w.astype(np.float64)
