from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

from common.im import im
from ops.modal_uhd_runner_fields import EarlyRejectFields, RunFields

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_DIRS = ("ops", "optimizer", "problems", "common", "sampling", "embedding")
_ENN_ROOT = _PROJECT_ROOT.parents[0] / "enn"


def _build_image(modal):
    image = (
        modal.Image.debian_slim(python_version="3.11.9")
        .apt_install("swig", "curl", "build-essential", "libopenblas-dev", "patchelf")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            'echo "export PATH=$HOME/.cargo/bin:$PATH" >> ~/.bashrc',
        )
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
            "maturin>=1.0",
        )
        .env({"PYTHONPATH": "/root"})
    )
    for project_dir in _PROJECT_DIRS:
        image = image.add_local_dir(str(_PROJECT_ROOT / project_dir), remote_path=f"/root/{project_dir}")
    image = image.add_local_dir(str(_ENN_ROOT), remote_path="/root/enn")
    return image.run_commands(
        ". $HOME/.cargo/env && "
        "export RUSTFLAGS='-C link-arg=-Wl,--no-as-needed -C link-arg=-lopenblas' && "
        "cd /root/enn/rust/crates/enn-py && maturin build --release",
        "pip install $(find /root/enn/rust -path '*/wheels/*.whl' | head -1) && pip install -e /root/enn",
    )


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


@dataclass(frozen=True)
class _ENNFields:
    minus_impute: bool
    d: int
    s: int
    jl_seed: int
    k: int
    fit_interval: int
    warmup_real_obs: int
    refresh_interval: int
    se_threshold: float
    target: str
    num_candidates: int
    select_interval: int
    embed_cfg: tuple[str, int]
    err_ema_beta: float = 0.95
    max_abs_err_ema: float = 0.25
    min_calib_points: int = 10

    @property
    def embedder(self) -> str:
        return self.embed_cfg[0]

    @property
    def gather_t(self) -> int:
        return self.embed_cfg[1]


def _run_fields(
    env_tag,
    num_rounds,
    policy_tag,
    lr,
    sigma,
    ndt,
    nmt,
    problem_seed,
    noise_seed_0,
    log_interval,
    accuracy_interval,
    target_accuracy,
) -> RunFields:
    return RunFields(
        env_tag=env_tag,
        num_rounds=int(num_rounds),
        policy_tag=policy_tag,
        lr=float(lr),
        sigma=float(sigma),
        ndt=ndt,
        nmt=nmt,
        problem_seed=None if problem_seed is None else int(problem_seed),
        noise_seed_0=None if noise_seed_0 is None else int(noise_seed_0),
        log_interval=int(log_interval),
        accuracy_interval=int(accuracy_interval),
        target_accuracy=target_accuracy,
    )


def _early_reject_fields(er) -> EarlyRejectFields:
    return EarlyRejectFields(
        tau=er.tau,
        mode=er.mode,
        ema_beta=er.ema_beta,
        warmup_pos=er.warmup_pos,
        quantile=er.quantile,
        window=er.window,
    )


def _parse_enn_fields(enn: dict[str, object] | None) -> _ENNFields:
    if hasattr(enn, "__dataclass_fields__"):
        enn = {
            "enn_minus_impute": enn.minus_impute,
            "enn_d": enn.d,
            "enn_s": enn.s,
            "enn_jl_seed": enn.jl_seed,
            "enn_k": enn.k,
            "enn_fit_interval": enn.fit_interval,
            "enn_warmup_real_obs": enn.warmup_real_obs,
            "enn_refresh_interval": enn.refresh_interval,
            "enn_se_threshold": enn.se_threshold,
            "enn_target": enn.target,
            "enn_num_candidates": enn.num_candidates,
            "enn_select_interval": enn.select_interval,
            "enn_embedder": enn.embedder,
            "enn_gather_t": enn.gather_t,
            "enn_err_ema_beta": enn.err_ema_beta,
            "enn_max_abs_err_ema": enn.max_abs_err_ema,
            "enn_min_calib_points": enn.min_calib_points,
        }
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
    enn_err_ema_beta = float(enn.get("enn_err_ema_beta", 0.95))
    enn_max_abs_err_ema = float(enn.get("enn_max_abs_err_ema", 0.25))
    enn_min_calib_points = int(enn.get("enn_min_calib_points", 10))
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
        err_ema_beta=enn_err_ema_beta,
        max_abs_err_ema=enn_max_abs_err_ema,
        min_calib_points=enn_min_calib_points,
    )


def run(
    env_tag,
    num_rounds,
    lr,
    ndt,
    nmt,
    *,
    gpu="A100",
    sigma: float = 0.001,
    policy_tag: str | None = None,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    log_interval: int = 10,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    early_reject=None,
    enn: dict[str, object] | None = None,
):
    modal = im("modal")
    EarlyRejectConfig = im("ops.uhd_config").EarlyRejectConfig
    make_loop = im("ops.uhd_setup_make_loop").make_loop

    app = modal.App(name="yubo-uhd")
    image = _build_image(modal)

    @app.function(image=image, timeout=2 * 60 * 60, gpu=gpu, serialized=True)
    def _run_uhd(run_f, er_f, enn_f):
        er_cfg = EarlyRejectConfig(
            tau=er_f.tau,
            mode=er_f.mode,
            ema_beta=er_f.ema_beta,
            warmup_pos=er_f.warmup_pos,
            quantile=er_f.quantile,
            window=er_f.window,
        )
        loop = make_loop(
            run_f.env_tag,
            run_f.num_rounds,
            policy_tag=run_f.policy_tag,
            problem_seed=run_f.problem_seed,
            noise_seed_0=run_f.noise_seed_0,
            lr=run_f.lr,
            sigma=run_f.sigma,
            num_dim_target=run_f.ndt,
            num_module_target=run_f.nmt,
            log_interval=run_f.log_interval,
            accuracy_interval=run_f.accuracy_interval,
            target_accuracy=run_f.target_accuracy,
            early_reject=er_cfg,
            enn={
                "enn_minus_impute": enn_f.minus_impute,
                "enn_d": enn_f.d,
                "enn_s": enn_f.s,
                "enn_jl_seed": enn_f.jl_seed,
                "enn_k": enn_f.k,
                "enn_fit_interval": enn_f.fit_interval,
                "enn_warmup_real_obs": enn_f.warmup_real_obs,
                "enn_refresh_interval": enn_f.refresh_interval,
                "enn_se_threshold": enn_f.se_threshold,
                "enn_target": enn_f.target,
                "enn_num_candidates": enn_f.num_candidates,
                "enn_select_interval": enn_f.select_interval,
                "enn_embedder": enn_f.embedder,
                "enn_gather_t": enn_f.gather_t,
                "enn_err_ema_beta": enn_f.err_ema_beta,
                "enn_max_abs_err_ema": enn_f.max_abs_err_ema,
                "enn_min_calib_points": enn_f.min_calib_points,
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

    er = (
        early_reject
        if early_reject is not None
        else EarlyRejectConfig(
            tau=None,
            mode=None,
            ema_beta=None,
            warmup_pos=None,
            quantile=None,
            window=None,
        )
    )
    with modal.enable_output():
        with app.run():
            return _run_uhd.remote(
                _run_fields(
                    env_tag,
                    num_rounds,
                    policy_tag,
                    lr,
                    sigma,
                    ndt,
                    nmt,
                    problem_seed,
                    noise_seed_0,
                    log_interval,
                    accuracy_interval,
                    target_accuracy,
                ),
                _early_reject_fields(er),
                enn_f,
            )
