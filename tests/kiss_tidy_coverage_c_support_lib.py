from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from kiss_problem_dm_fake import _FakeDM


def _init_enn_imputer(Comb, m, cfg, _nz, attrs):
    imp = Comb()
    imp._module = m
    imp._cfg = cfg
    imp._noise_nz_fn = _nz
    imp._z_base = torch.zeros(cfg.d, dtype=torch.float32)
    for a, v in attrs:
        setattr(imp, a, v)
    return imp


def _enn_posterior_model():
    class _Post:
        def __init__(self, mu, se):
            self.mu = np.array(mu)
            self.se = np.array(se)

    class _EM:
        def posterior(self, x, params=None, flags=None):
            n = int(np.asarray(x).shape[0])
            return _Post([0.1] * n, [0.01] * n)

    return _EM()


def _run_uhd_enn_gather_imputer(Comb, m, cfg, _nz):
    from sampling.gather_proj_t import GatherProjSpec

    da = int(sum(p.numel() for p in m.parameters()))
    attrs = [
        ("_delta_z", None),
        ("_delta_x", None),
        ("_x", []),
        ("_y", []),
        ("_num_new_since_fit", 0),
        ("_num_real_evals", 0),
        ("_num_imputed", 0),
        ("_enn_model", None),
        ("_enn_params", None),
        ("_num_negative_phases", 0),
        ("_last_mu_plus", None),
        ("_num_calib", 0),
        ("_abs_err_ema", None),
        ("_gather_spec", GatherProjSpec.make(dim_ambient=da, d=8, t=64, seed=1)),
    ]
    imp = _init_enn_imputer(Comb, m, cfg, _nz, attrs)
    imp.begin_pair(seed=0, sigma=0.1)
    imp.tell_real(mu=0.5, phase="plus")
    imp.tell_real(mu=0.25, phase="minus")
    imp.choose_seed_ucb(base_seed=0, sigma=0.1)
    imp.update_base_after_step(step_scale=1.0, sigma=0.1)


def _run_uhd_enn_direction_imputer(Comb, m, cfgd, _nz, em):
    attrs = [
        ("_delta_z", None),
        ("_delta_x", None),
        ("_x", []),
        ("_y", []),
        ("_num_new_since_fit", 0),
        ("_num_real_evals", 0),
        ("_num_imputed", 0),
        ("_enn_model", None),
        ("_enn_params", None),
        ("_num_negative_phases", 0),
        ("_last_mu_plus", None),
        ("_gather_spec", None),
        ("_num_calib", 10),
        ("_abs_err_ema", 0.01),
    ]
    imp2 = _init_enn_imputer(Comb, m, cfgd, _nz, attrs)
    imp2.begin_pair(seed=1, sigma=0.1)
    imp2.tell_real(mu=1.0, phase="plus")
    imp2.tell_real(mu=0.5, phase="minus")
    imp2.choose_seed_ucb(base_seed=0, sigma=0.1)
    imp2._enn_model = em
    imp2._enn_params = object()
    imp2._y_mean = 0.0
    imp2._y_std = 1.0
    imp2._last_mu_plus = 1.0
    imp2.predict_current()
    imp2.try_impute_current()
    imp3 = _init_enn_imputer(
        Comb,
        m,
        cfgd,
        _nz,
        [
            ("_delta_z", torch.zeros(16, dtype=torch.float32)),
            ("_delta_x", np.zeros(16, dtype=np.float64)),
            ("_x", []),
            ("_y", []),
            ("_num_new_since_fit", 0),
            ("_num_real_evals", 0),
            ("_num_imputed", 0),
            ("_enn_model", em),
            ("_enn_params", object()),
            ("_y_mean", 0.0),
            ("_y_std", 1.0),
            ("_last_mu_plus", None),
            ("_gather_spec", None),
            ("_num_calib", 0),
            ("_abs_err_ema", 0.0),
            ("_num_negative_phases", 0),
        ],
    )
    with pytest.raises(RuntimeError):
        imp3.predict_current()


def run_uhd_enn_mixins(_monkeypatch):
    from optimizer.uhd_enn_config import ENNImputerConfig, _compute_z
    from optimizer.uhd_enn_imputer_predict import ENNMinusImputerPredictMixin
    from optimizer.uhd_enn_imputer_tell import ENNMinusImputerTellMixin

    assert ENNMinusImputerPredictMixin and ENNMinusImputerTellMixin
    z0 = torch.zeros(4, dtype=torch.float32)
    assert _compute_z(z0, torch.ones(4, dtype=torch.float32), 1.0).shape == (4,)
    Comb = type("Comb", (ENNMinusImputerTellMixin, ENNMinusImputerPredictMixin), {})
    m = nn.Linear(3, 1, bias=True)

    def _nz(_s, _sig):
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float32)

    cfg = ENNImputerConfig(
        d=8,
        s=2,
        warmup_real_obs=1,
        fit_interval=10**9,
        k=3,
        embedder="gather",
        target="mu_minus",
        min_calib_points=0,
    )
    _run_uhd_enn_gather_imputer(Comb, m, cfg, _nz)
    cfgd = ENNImputerConfig(
        d=16,
        s=2,
        warmup_real_obs=0,
        fit_interval=1,
        k=3,
        num_candidates=3,
        target="delta",
        min_calib_points=0,
        embedder="direction",
    )
    _run_uhd_enn_direction_imputer(Comb, m, cfgd, _nz, _enn_posterior_model())


def _make_uhd_loop_with_early_reject():
    from optimizer.uhd_loop_support import UHDLoopSupportMixin

    Loop = type("Loop", (UHDLoopSupportMixin,), {})
    lo = Loop()
    lo._log_interval = 2
    lo._accuracy_fn = lambda: 0.25
    lo._accuracy_interval = 2
    lo._log_param_stats = True
    lo._module = nn.Linear(3, 2, bias=True)
    lo._enn_minus_imputer = SimpleNamespace(abs_err_ema=0.1, num_real_evals=1, num_imputed=0, num_candidates=2)
    lo._enn_seed_selector = object()
    lo._enn_num_selected = 1
    lo._early_reject_tau = 0.5
    lo._early_reject_mode = "y_best"
    lo._early_reject_warmup_pos = 1
    lo._early_reject_num_pos_seen = 2
    lo._early_reject_mu_plus_ema = 1.0
    lo._early_reject_ema_beta = 0.9
    lo._early_reject_window = 3
    lo._early_reject_mu_plus_window = [1.0, 2.0, 0.5]
    lo._early_reject_quantile = 0.5
    lo._early_reject_skips = 0
    lo._uhd = SimpleNamespace(sigma=0.1, mu_avg=0.0, se_avg=0.1, y_best=1.0)
    return lo


def _run_uhd_loop_logging_paths(lo):
    assert lo._should_log(i_iter=0, last_iter=5) and lo._should_log_enn_stats(i_iter=0, last_iter=5)
    m1, s1 = lo._maybe_compute_param_stats()
    assert m1 is not None
    assert lo._maybe_update_accuracy(i_iter=4, last_iter=5, acc=None) is not None
    assert "EVAL" in lo._format_eval_line(
        i_iter=0,
        proposal_dt=0.0001,
        eval_dt=0.0002,
        y_best_str="1",
        mu=0.0,
        se=0.1,
        acc=0.2,
        mean_param=m1,
        std_param=s1,
    )
    lo._print_log_block(
        i_iter=0,
        last_iter=5,
        proposal_dt=0.0001,
        eval_dt=0.0002,
        y_best_str="1",
        acc=0.1,
        mean_param=m1,
        std_param=s1,
    )


def _run_uhd_loop_early_reject_modes(lo):
    assert not lo._should_early_reject(mu_plus=2.0)
    lo._early_reject_mode = "ema"
    assert not lo._should_early_reject(mu_plus=2.0)
    lo._early_reject_mode = "quantile"
    assert lo._should_early_reject(mu_plus=-1.0) in (True, False)
    lo._update_early_reject_state(mu_plus=0.5)
    lo._enn_minus_imputer = None
    lo._print_log_block(
        i_iter=0,
        last_iter=1,
        proposal_dt=0.0,
        eval_dt=0.0,
        y_best_str="1",
        acc=None,
        mean_param=None,
        std_param=None,
    )
    lo._early_reject_mode = "bad"
    with pytest.raises(ValueError):
        lo._should_early_reject(mu_plus=0.0)


def run_uhd_loop_support(capsys):
    from optimizer.uhd_loop_support import UHDLoopSupportMixin

    lo = _make_uhd_loop_with_early_reject()
    _run_uhd_loop_logging_paths(lo)
    _run_uhd_loop_early_reject_modes(lo)
    Loop = type("Loop", (UHDLoopSupportMixin,), {})
    lo2 = Loop()
    lo2._log_interval = 5
    lo2._accuracy_fn = None
    lo2._accuracy_interval = 5
    lo2._log_param_stats = False
    lo2._module = nn.Module()
    lo2._enn_minus_imputer = None
    lo2._enn_seed_selector = None
    lo2._enn_num_selected = 0
    lo2._early_reject_tau = None
    lo2._early_reject_mode = "ema"
    lo2._early_reject_warmup_pos = 10
    lo2._early_reject_num_pos_seen = 0
    lo2._early_reject_mu_plus_ema = None
    lo2._early_reject_ema_beta = 0.5
    lo2._early_reject_window = 0
    lo2._early_reject_mu_plus_window = []
    lo2._early_reject_quantile = 0.5
    lo2._uhd = SimpleNamespace(sigma=0.2, mu_avg=0.1, se_avg=0.2, y_best=None)
    assert lo2._maybe_compute_param_stats() == (None, None)
    lo2._maybe_update_accuracy(i_iter=1, last_iter=10, acc=0.1)
    _ = capsys.readouterr()


def _run_dm_control_core_paths(monkeypatch):
    import problems.dm_control_env_core as dce

    monkeypatch.setattr(dce, "suite", types.SimpleNamespace(load=lambda *a, **k: _FakeDM()))
    monkeypatch.setattr(sys, "platform", "linux")
    for e in ("MUJOCO_GL", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(e, raising=False)
    dce.configure_headless_render_backend("rgb_array")
    dce.configure_headless_render_backend(None)
    assert dce.parse_env_name("dm_control/cheetah-run-v0") == ("cheetah", "run")
    with pytest.raises(ValueError):
        dce.parse_env_name("pendulum-v1")
    with pytest.raises(ValueError):
        dce.parse_env_name("dm_control/cheetahrun-v0")
    env = dce.DMControlEnv("cheetah", "run", render_mode=None, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] >= 1
    env.step(np.zeros((2,), dtype=np.float32))
    env.close()
    return dce


def _run_dm_pixel_and_env_conf_paths(monkeypatch, dce):
    from problems.dm_control_pixel_wrapper import PixelObsWrapper, make_dm_control
    from problems.dm_control_spaces import (
        BoxSpace,
        DictSpace,
        flatten_obs,
        is_gl_init_error,
        resize_pixels,
        spec_bounds,
        spec_to_space,
    )
    from problems.env_conf_backends import (
        maybe_register_atari_dm_backends,
        register_with_env_conf,
    )
    from problems.env_conf_bindings import (
        get_atari_dm_bindings,
        register_atari_dm_bindings_loader,
    )
    from problems.env_conf_parse import parse_tag_options
    from problems.env_conf_rl import resolve_rl_model_defaults
    from problems.env_conf_types import GymConf
    from problems.pixel_policies_encoders import (
        init_linear_and_conv,
        nature_cnn_encoder,
        obs_space_from_env_conf,
        tiny_atari_cnn_encoder,
    )

    fe = dce.DMControlEnv("cheetah", "run", render_mode="rgb_array", seed=0)
    monkeypatch.setattr(fe, "_env", _FakeDM())
    pw = PixelObsWrapper(fe, pixels_only=True, size=84)
    assert pw.reset(seed=0)[0].shape == (84, 84, 3)
    pw2 = PixelObsWrapper(fe, pixels_only=False, size=84)
    assert "pixels" in pw2.reset(seed=0)[0]
    pw.close()
    pw2.close()
    e0 = make_dm_control("dm_control/cheetah-run-v0", from_pixels=False)
    e1 = make_dm_control("dm_control/cheetah-run-v0", from_pixels=True, pixels_only=True)
    e0.close()
    e1.close()
    sp = SimpleNamespace(shape=(2,), minimum=np.zeros(2), maximum=np.ones(2))
    assert spec_bounds(sp)[0].shape == (2,)
    sp2 = SimpleNamespace(shape=(1,), minimum=None, maximum=None)
    assert np.isneginf(spec_bounds(sp2)[0][0])
    dspec = {"a": sp, "b": sp2}
    assert isinstance(spec_to_space(dspec), BoxSpace)
    assert "u" in DictSpace({"u": BoxSpace(np.zeros(1), np.ones(1))}).sample()
    assert flatten_obs({"b": np.zeros(1), "a": np.zeros(2)}).shape[0] == 3
    assert flatten_obs({}).shape == (0,)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert resize_pixels(img, 10, 10).shape == (10, 10, 3)
    assert resize_pixels(img, 5, 5).shape == (5, 5, 3)
    assert is_gl_init_error(Exception("gladLoadGL failed")) and not is_gl_init_error(Exception("other"))
    maybe_register_atari_dm_backends("pend")
    maybe_register_atari_dm_backends("dm:cheetah-run")
    register_atari_dm_bindings_loader(lambda: SimpleNamespace())
    assert get_atari_dm_bindings() is not None
    register_with_env_conf()
    _t, fn, fp = parse_tag_options("a:b:fn:pixels", None)
    assert fn and fp
    pend, fn2, _fp2 = parse_tag_options("pend", True)
    assert pend == "pend" and not fn2
    assert isinstance(resolve_rl_model_defaults("cheetah", algo="ppo"), dict)
    assert GymConf(max_steps=10).max_steps == 10
    with pytest.raises(ValueError):
        obs_space_from_env_conf(SimpleNamespace(state_space=None))
    bx = BoxSpace(np.zeros(2), np.ones(2))
    assert obs_space_from_env_conf(SimpleNamespace(state_space=bx)) is bx
    seq = nn.Sequential(nn.Linear(2, 2), nn.Conv2d(1, 1, 1))
    init_linear_and_conv(seq, gain=0.1)
    enc1, d1 = tiny_atari_cnn_encoder(4)
    assert enc1(torch.zeros(1, 4, 84, 84)).shape[-1] == d1
    enc2, d2 = nature_cnn_encoder(3, latent_dim=128)
    _ = enc2(torch.zeros(1, 3, 84, 84))
    assert d2 == 128


def run_dm_pixel_env_conf(monkeypatch):
    dce = _run_dm_control_core_paths(monkeypatch)
    _run_dm_pixel_and_env_conf_paths(monkeypatch, dce)
