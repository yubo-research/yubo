from pathlib import Path

import modal

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_DIRS = ("ops", "optimizer", "problems", "common", "sampling", "embedding")

_image = (
    modal.Image.debian_slim(python_version="3.11.9")
    .apt_install("swig")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.4",
        "gymnasium==1.2.0",
        "gymnasium[mujoco]",
        "gymnasium[box2d]",
        "mujoco==3.3.3",
        "scipy==1.15.3",
        "click==8.3.1",
    )
    .env({"PYTHONPATH": "/root"})
)
for _d in _PROJECT_DIRS:
    _image = _image.add_local_dir(str(_PROJECT_ROOT / _d), remote_path=f"/root/{_d}")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


class _ENNFields:
    def __init__(
        self,
        *,
        minus_impute: bool,
        d: int,
        s: int,
        jl_seed: int,
        k: int,
        fit_interval: int,
        warmup_real_obs: int,
        refresh_interval: int,
        se_threshold: float,
        target: str,
        num_candidates: int,
        select_interval: int,
        embed_cfg: tuple[str, int],
    ):
        self.minus_impute = minus_impute
        self.d = d
        self.s = s
        self.jl_seed = jl_seed
        self.k = k
        self.fit_interval = fit_interval
        self.warmup_real_obs = warmup_real_obs
        self.refresh_interval = refresh_interval
        self.se_threshold = se_threshold
        self.target = target
        self.num_candidates = num_candidates
        self.select_interval = select_interval
        self.embedder, self.gather_t = embed_cfg


def _parse_enn_fields(enn: dict[str, object] | None) -> _ENNFields:
    enn = {} if enn is None else dict(enn)
    enn_minus_impute = bool(enn.get("enn_minus_impute", False))
    enn_d = int(enn.get("enn_d", 100))
    enn_s = int(enn.get("enn_s", 4))
    enn_jl_seed = int(enn.get("enn_jl_seed", 123))
    enn_k = int(enn.get("enn_k", 25))
    enn_fit_interval = int(enn.get("enn_fit_interval", 50))
    enn_warmup_real_obs = int(enn.get("enn_warmup_real_obs", 200))
    enn_refresh_interval = int(enn.get("enn_refresh_interval", 50))
    enn_se_threshold = float(enn.get("enn_se_threshold", 0.25))
    enn_target = str(enn.get("enn_target", "mu_minus"))
    enn_num_candidates = int(enn.get("enn_num_candidates", 1))
    enn_select_interval = int(enn.get("enn_select_interval", 1))
    enn_embedder = str(enn.get("enn_embedder", "direction"))
    enn_gather_t = int(enn.get("enn_gather_t", 64))
    return _ENNFields(
        minus_impute=enn_minus_impute,
        d=enn_d,
        s=enn_s,
        jl_seed=enn_jl_seed,
        k=enn_k,
        fit_interval=enn_fit_interval,
        warmup_real_obs=enn_warmup_real_obs,
        refresh_interval=enn_refresh_interval,
        se_threshold=enn_se_threshold,
        target=enn_target,
        num_candidates=enn_num_candidates,
        select_interval=enn_select_interval,
        embed_cfg=(enn_embedder, enn_gather_t),
    )


def _run(
    env_tag,
    num_rounds,
    lr,
    ndt,
    nmt,
    *,
    gpu="A100",
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    log_interval: int = 10,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    early_reject_tau: float | None = None,
    early_reject_mode: str | None = None,
    early_reject_ema_beta: float | None = None,
    early_reject_warmup_pos: int | None = None,
    early_reject_quantile: float | None = None,
    early_reject_window: int | None = None,
    enn: dict[str, object] | None = None,
):
    app = modal.App(name="yubo-uhd")

    @app.function(image=_image, timeout=2 * 60 * 60, gpu=gpu, serialized=True)
    def _run_uhd(
        env_tag,
        num_rounds,
        lr,
        sigma,
        ndt,
        nmt,
        problem_seed,
        noise_seed_0,
        log_interval,
        accuracy_interval,
        target_accuracy,
        early_reject_tau,
        early_reject_mode,
        early_reject_ema_beta,
        early_reject_warmup_pos,
        early_reject_quantile,
        early_reject_window,
        enn_minus_impute,
        enn_d,
        enn_s,
        enn_jl_seed,
        enn_k,
        enn_fit_interval,
        enn_warmup_real_obs,
        enn_refresh_interval,
        enn_se_threshold,
        enn_target,
        enn_num_candidates,
        enn_select_interval,
        enn_embedder,
        enn_gather_t,
    ):
        import io
        import sys

        from ops.uhd_setup import make_loop

        loop = make_loop(
            env_tag,
            num_rounds,
            problem_seed=problem_seed,
            noise_seed_0=noise_seed_0,
            lr=lr,
            sigma=sigma,
            num_dim_target=ndt,
            num_module_target=nmt,
            log_interval=log_interval,
            accuracy_interval=accuracy_interval,
            target_accuracy=target_accuracy,
            early_reject_tau=early_reject_tau,
            early_reject_mode=early_reject_mode,
            early_reject_ema_beta=early_reject_ema_beta,
            early_reject_warmup_pos=early_reject_warmup_pos,
            early_reject_quantile=early_reject_quantile,
            early_reject_window=early_reject_window,
            enn={
                "enn_minus_impute": enn_minus_impute,
                "enn_d": enn_d,
                "enn_s": enn_s,
                "enn_jl_seed": enn_jl_seed,
                "enn_k": enn_k,
                "enn_fit_interval": enn_fit_interval,
                "enn_warmup_real_obs": enn_warmup_real_obs,
                "enn_refresh_interval": enn_refresh_interval,
                "enn_se_threshold": enn_se_threshold,
                "enn_target": enn_target,
                "enn_num_candidates": enn_num_candidates,
                "enn_select_interval": enn_select_interval,
                "enn_embedder": enn_embedder,
                "enn_gather_t": enn_gather_t,
            },
        )
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, buf)
        try:
            loop.run()
        finally:
            sys.stdout = real_stdout
        return buf.getvalue()

    enn_f = _parse_enn_fields(enn)

    with modal.enable_output():
        with app.run():
            return _run_uhd.remote(
                env_tag,
                num_rounds,
                lr,
                0.001,
                ndt,
                nmt,
                None if problem_seed is None else int(problem_seed),
                None if noise_seed_0 is None else int(noise_seed_0),
                int(log_interval),
                int(accuracy_interval),
                target_accuracy,
                None if early_reject_tau is None else float(early_reject_tau),
                None if early_reject_mode is None else str(early_reject_mode),
                None if early_reject_ema_beta is None else float(early_reject_ema_beta),
                None if early_reject_warmup_pos is None else int(early_reject_warmup_pos),
                None if early_reject_quantile is None else float(early_reject_quantile),
                None if early_reject_window is None else int(early_reject_window),
                bool(enn_f.minus_impute),
                int(enn_f.d),
                int(enn_f.s),
                int(enn_f.jl_seed),
                int(enn_f.k),
                int(enn_f.fit_interval),
                int(enn_f.warmup_real_obs),
                int(enn_f.refresh_interval),
                float(enn_f.se_threshold),
                str(enn_f.target),
                int(enn_f.num_candidates),
                int(enn_f.select_interval),
                str(enn_f.embedder),
                int(enn_f.gather_t),
            )


run = _run
