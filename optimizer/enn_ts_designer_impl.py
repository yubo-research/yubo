"""ENN Thompson sampling via per-arm TURBO-0 maximization of a fixed function draw."""

from __future__ import annotations

import tempfile
import time


def _im(name: str):
    return __import__(name, fromlist=["_"])


class EnnTSDesigner:
    def __init__(
        self,
        policy,
        *,
        num_ts_iters: int,
        k: int = 10,
        num_fit_samples: int = 100,
        num_fit_candidates: int = 100,
        num_init: int | None = None,
        index_driver: str | None = None,
    ):
        if int(num_ts_iters) < 0:
            raise ValueError(f"num_ts_iters must be >= 0, got {num_ts_iters}")
        if num_init is not None and int(num_init) < 2:
            raise ValueError(f"num_init must be >= 2, got {num_init}")
        self._policy = policy
        self._num_ts_iters = int(num_ts_iters)
        self._k = int(k)
        self._num_fit_samples = int(num_fit_samples)
        self._num_fit_candidates = int(num_fit_candidates)
        self._num_init = None if num_init is None else int(num_init)
        self._index_driver = index_driver
        self._enn_disk_work_dir = None

        np = _im("numpy")
        self._rng = np.random.default_rng(np.random.randint(2**31))
        self._num_told = 0
        self._enn = None
        self._params = None
        self._bounds = None

    def _resolve_index_driver(self):
        return _im("optimizer.enn_index_driver_util").resolve_enn_index_driver(self._index_driver)

    def _enn_disk_kwargs(self, index_driver):
        ENNIndexDriver = _im("enn.turbo.config.enn_index_driver").ENNIndexDriver
        if index_driver is not ENNIndexDriver.BPANN_DISK:
            return {}
        if self._enn_disk_work_dir is None:
            self._enn_disk_work_dir = tempfile.TemporaryDirectory(prefix="yubo_enn_ts_bpann_disk_")
        return {
            "enn_storage": "disk",
            "work_dir": self._enn_disk_work_dir.name,
        }

    def _ensure_enn(self):
        if self._enn is not None:
            return
        np = _im("numpy")
        EpistemicNearestNeighbors = _im("enn.enn.enn_class").EpistemicNearestNeighbors
        num_dim = int(self._policy.num_params())
        index_driver = self._resolve_index_driver()
        self._enn = EpistemicNearestNeighbors(
            np.empty((0, num_dim), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
            None,
            scale_x=False,
            index_driver=index_driver,
            **self._enn_disk_kwargs(index_driver),
        )
        ab = _im("common.all_bounds")
        self._bounds = np.array([[ab.x_low, ab.x_high]] * num_dim, dtype=np.float64)

    def _default_params(self):
        ENNParams = _im("enn.enn.enn_params").ENNParams
        k = self._k
        if self._enn is not None and len(self._enn) > 0:
            k = min(self._k, max(1, len(self._enn) - 1)) if len(self._enn) > 1 else 1
        return ENNParams(
            k_num_neighbors=k,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        )

    def _append_new_data(self, new_data):
        if len(new_data) == 0:
            return
        da = _im("optimizer.designer_asserts")
        da.assert_scalar_rreturn(new_data)
        np = _im("numpy")
        self._ensure_enn()
        x = np.asarray([d.policy.get_params() for d in new_data], dtype=np.float64)
        y = np.asarray([float(d.trajectory.rreturn) for d in new_data], dtype=np.float64).reshape(-1, 1)
        self._enn.add(x, y)

    def _fit_params(self):
        self._ensure_enn()
        if len(self._enn) == 0:
            self._params = self._default_params()
            return
        ENNStatefulFitter = _im("enn.enn.enn_fitter").ENNStatefulFitter
        np = _im("numpy")
        idx = list(range(len(self._enn)))
        x_all, y_all, yvar = self._enn.train_rows_at(idx)
        fitter = ENNStatefulFitter(
            k=self._k,
            rng=self._rng,
            infer_aleatoric_variance_scale=True,
        )
        fitter.tell(np.asarray(x_all, dtype=np.float64), y_all, yvar)
        self._params = fitter.ask(
            self._enn,
            num_fit_candidates=self._num_fit_candidates,
            num_fit_samples=self._num_fit_samples,
            params_warm_start=self._params,
        )

    def _draw_values(self, x, seed_ts):
        PosteriorFlags = _im("enn.enn.enn_params").PosteriorFlags
        np = _im("numpy")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        draws, _idx = self._enn.posterior_function_draw(
            x,
            self._params,
            function_seeds=[int(seed_ts)],
            flags=PosteriorFlags(observation_noise=False),
        )
        # draws: (batch, metrics, num_seeds) → (batch, metrics)
        return np.asarray(draws[:, :, 0], dtype=np.float64)

    def _build_init_X(self, seed_ts, num_init: int):
        np = _im("numpy")
        draw_lhd = _im("enn.turbo.python_fallback.sampling").draw_lhd
        n_lhd = num_init // 2
        assert n_lhd >= 1
        x_lhd = draw_lhd(bounds=self._bounds, num_arms=n_lhd, rng=self._rng)
        n_obs = 0 if self._enn is None else len(self._enn)
        n_inc = min(n_obs, n_lhd)
        if n_inc == 0:
            return np.asarray(x_lhd, dtype=np.float64)
        x_obs, _y_obs, _yvar = self._enn.train_rows_at(list(range(n_obs)))
        x_obs = np.asarray(x_obs, dtype=np.float64)
        y_draw = self._draw_values(x_obs, seed_ts).reshape(-1)
        top_idx = np.argsort(-y_draw)[:n_inc]
        x_inc = x_obs[top_idx]
        return np.concatenate([x_lhd, x_inc], axis=0)

    def _maximize_draw(self, seed_ts):
        np = _im("numpy")
        enn = _im("enn")
        factory = _im("enn.turbo.config.factory")
        num_init = self._num_init
        assert num_init is not None and num_init >= 2
        init_X = self._build_init_X(seed_ts, num_init)
        init_y = self._draw_values(init_X, seed_ts)
        cfg = factory.turbo_zero_config(num_init=len(init_X))
        opt = enn.create_optimizer(bounds=self._bounds, config=cfg, rng=self._rng)
        opt.tell(init_X, init_y)

        best_x = np.asarray(init_X, dtype=np.float64)
        best_y = np.asarray(init_y, dtype=np.float64).reshape(-1)
        best_i = int(np.argmax(best_y))
        best_val = float(best_y[best_i])
        best_point = best_x[best_i].copy()

        for _ in range(self._num_ts_iters):
            x_ask = opt.ask(1)
            y_ask = self._draw_values(x_ask, seed_ts)
            opt.tell(x_ask, y_ask)
            y_flat = np.asarray(y_ask, dtype=np.float64).reshape(-1)
            assert y_flat.size == 1
            if float(y_flat[0]) > best_val:
                best_val = float(y_flat[0])
                best_point = np.asarray(x_ask, dtype=np.float64).reshape(-1).copy()
        return best_point

    def _make_policy(self, x):
        policy = self._policy.clone()
        policy.set_params(x)
        return policy

    def __call__(self, data, num_arms, *, telemetry=None):
        if self._num_init is None:
            self._num_init = max(2, int(num_arms))
        self._ensure_enn()

        if data is not None and len(data) > self._num_told:
            new_data = data[self._num_told :]
            self._append_new_data(new_data)
            self._num_told = len(data)

        t0 = time.perf_counter()
        self._fit_params()
        dt_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        policies = []
        for _ in range(int(num_arms)):
            seed_ts = int(self._rng.integers(0, 2**31))
            x_star = self._maximize_draw(seed_ts)
            policies.append(self._make_policy(x_star))
        dt_select = time.perf_counter() - t1

        if telemetry is not None:
            telemetry.set_dt_fit(dt_fit)
            telemetry.set_dt_select(dt_select)
        return policies
