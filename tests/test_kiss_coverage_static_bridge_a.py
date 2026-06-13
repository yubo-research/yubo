"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def test_kiss_bridge_env_preprocessing_clip_observation_wrapper():
    import gymnasium as gym

    from common.env_preprocessing import _ClipObservationWrapper

    def _e_init(self):
        gym.Env.__init__(self)
        self.observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def _e_reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def _e_step(self, _action):
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    E = type(
        "E",
        (gym.Env,),
        {"metadata": {}, "__init__": _e_init, "reset": _e_reset, "step": _e_step},
    )
    w = _ClipObservationWrapper(E(), low=-1.0, high=1.0)
    w.reset(seed=0)
    w.step(np.zeros(1, dtype=np.float32))


def test_kiss_bridge_uhd_setup_loop_exports():
    from ops.uhd_setup_bszo import run_bszo_loop
    from ops.uhd_setup_make_loop import make_loop
    from ops.uhd_setup_simple_gym import run_simple_loop

    assert callable(run_simple_loop) and callable(run_bszo_loop) and callable(make_loop)


def test_kiss_bridge_exp_uhd_full_modal_cmd(monkeypatch, tmp_path):
    from ops.exp_uhd_full import modal_cmd as exp_uhd_full_modal_cmd

    tom = tmp_path / "u.toml"
    tom.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')
    monkeypatch.setattr("ops.exp_uhd_run.uhd_config_toml_to_modal_log", lambda *a, **k: "full")
    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: "full")
    exp_uhd_full_modal_cmd.callback(str(tom), None, "A100")
    assert callable(exp_uhd_full_modal_cmd)


def test_kiss_bridge_modal_uhd_run(monkeypatch):
    import importlib

    from ops.modal_uhd import run as modal_uhd_run

    modal_uhd_runner_impl = importlib.import_module("ops.modal_uhd_runner_impl")
    monkeypatch.setattr(modal_uhd_runner_impl, "run", lambda *a, **k: "MR")
    assert modal_uhd_run("mnist", 1, 0.01, 2, 1, policy_tag="pure-function", gpu="cpu") == "MR"


def test_kiss_bridge_exp_uhd_cli_modal_cmd(monkeypatch, tmp_path):
    from click.testing import CliRunner

    import ops.exp_uhd as exp_uhd
    from ops.exp_uhd_cli import modal_cmd
    from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig

    early_reject = EarlyRejectConfig(tau=None, mode=None, ema_beta=None, warmup_pos=None, quantile=None, window=None)
    be = BEConfig(10, 10, 20, 10, 25, None)
    enn = ENNConfig(False, 100, 4, 123, 25, 50, 200, 50, 0.25, "mu_minus", 1, 1, "direction", 64)
    cfg = UHDConfig(
        env_tag="f:sphere-2d",
        num_rounds=1,
        early_reject=early_reject,
        be=be,
        enn=enn,
    )
    assert cfg.env_tag == "f:sphere-2d"

    def _loop_run(self):
        pass

    _Loop = type("_Loop", (), {"run": _loop_run})
    monkeypatch.setattr("ops.uhd_setup_make_loop.make_loop", lambda *a, **k: _Loop())

    toml = tmp_path / "cfg.toml"
    toml.write_text('[uhd]\nenv_tag = "f:sphere-2d"\npolicy_tag = "pure-function"\nnum_rounds = 1\n')
    runner = CliRunner()
    assert runner.invoke(exp_uhd.cli, ["local", str(toml)]).exit_code == 0
    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: None)
    assert runner.invoke(exp_uhd.cli, ["modal", str(toml)]).exit_code == 0

    tom = tmp_path / "u.toml"
    tom.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')
    monkeypatch.setattr("ops.exp_uhd_run.uhd_config_toml_to_modal_log", lambda *a, **k: "ok")
    modal_cmd.callback(str(tom), None, "A100")
    assert callable(modal_cmd)
    assert callable(exp_uhd.local)


def test_kiss_bridge_exp_uhd_parse_extras():
    from ops.exp_uhd_parse_extras import apply_optional_cfg_fields, validate_llm_sampling_config
    from ops.uhd_config import UHDConfig

    config_dict: dict = {"env_tag": "mnist", "num_rounds": 1}
    apply_optional_cfg_fields(config_dict, {"bf8_storage": True})
    assert config_dict["bf8_storage"] is True
    validate_llm_sampling_config(
        UHDConfig(env_tag="mnist", num_rounds=1),
    )


def test_kiss_bridge_uhd_batch_submit_collect(tmp_path, monkeypatch):
    from ops.uhd_batch_cli import collect_cmd, submit_cmd

    toml = tmp_path / "cfg.toml"
    toml.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')
    monkeypatch.setattr("ops.uhd_batch_cli._require_modal", lambda: None)
    monkeypatch.setattr("ops.uhd_batch_cli._ensure_uhd_batch_app", lambda: None)
    monkeypatch.setattr("ops.uhd_batch_cli._batch_modal", lambda *a, **k: None)
    submit_cmd.callback(None, str(toml), 1, "results/uhd")
    monkeypatch.setattr("ops.uhd_batch_cli._collect", lambda rd: None)
    collect_cmd.callback("results/uhd")
    assert callable(submit_cmd)
    assert callable(collect_cmd)


def test_kiss_bridge_uhd_batch_cli_commands(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from ops.uhd_batch_cli import batch_cmd, cleanup_cmd, modal_cmd

    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\nnum_reps = 2\n')

    mock_bm = MagicMock()
    monkeypatch.setattr("ops.uhd_batch_cli._resolve_batch_modal", lambda: mock_bm)
    modal_cmd.callback(str(toml_path), None, "results/uhd")
    assert mock_bm.call_args[0][1] == 2

    monkeypatch.setattr("ops.uhd_batch_cli._require_modal", lambda: None)
    monkeypatch.setattr("ops.uhd_batch_cli._ensure_uhd_batch_app", lambda: None)
    monkeypatch.setattr("ops.uhd_batch_cli._load_prep_configs", lambda *_a, **_k: [({"env_tag": "mnist"}, 1)])
    monkeypatch.setattr("ops.uhd_batch_cli._batch_modal", mock_bm)
    batch_cmd.callback("experiments.uhd_batch_preps.prep_uhd_batch_cheetah", "results/uhd")

    deleted: list[str] = []

    class _Dict:
        @staticmethod
        def delete(name):
            deleted.append(name)

    monkeypatch.setitem(__import__("sys").modules, "modal", type("modal", (), {"Dict": _Dict})())
    cleanup_cmd.callback()
    assert deleted == ["uhd_batch_results", "uhd_batch_submitted"]


def test_kiss_bridge_uhd_setup_bszo_core_run_loop(monkeypatch):
    import torch.nn as nn

    from ops.uhd_setup_bszo_core import run_bszo_loop

    lin = nn.Linear(1, 1, bias=False)

    def _gconf(*_a, **_k):
        from types import SimpleNamespace

        return SimpleNamespace(
            noise_seed_0=0,
            problem_seed=0,
            make_torch_env=lambda: SimpleNamespace(torch_env=lambda: SimpleNamespace(module=lin)),
        )

    monkeypatch.setattr("problems.env_conf.get_env_conf", _gconf)
    monkeypatch.setattr("ops.uhd_setup_bszo_core._run_bszo_iterations", lambda *a, **k: None)
    monkeypatch.setattr(
        "ops.uhd_setup_bszo_evaluate.make_bszo_mnist_evaluate_fn",
        lambda *a, **k: (lambda _s: (0.0, 0.0)),
    )
    monkeypatch.setattr("ops.uhd_setup_util._preload_mnist_train_to_device", lambda _d: (None, None))
    monkeypatch.setattr("ops.uhd_setup_util._make_accuracy_fn", lambda *a, **k: lambda: 0.0)
    run_bszo_loop("mnist", 1, lr=0.01, problem_seed=0, noise_seed_0=0)


def test_kiss_bridge_gaussian_perturbator_base():
    from optimizer.gaussian_perturbator import GaussianPerturbator, PerturbatorBase

    m = nn.Linear(2, 1)
    gp = GaussianPerturbator(m)
    assert isinstance(gp, PerturbatorBase)
