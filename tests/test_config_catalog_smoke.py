import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from common.config_toml import load_toml
from experiments.experiment import cli as bo_cli
from experiments.experiment import load_experiment_config
from rl import builtins as rl_builtins
from rl import runner as rl_runner

_REPO_ROOT = Path(__file__).resolve().parents[1]

_TARGET_BO_TOMLS = (
    "configs/bo/cheetah/exp_cheetah_natural_turbo_enn_p.toml",
    "configs/bo/dm_control/exp_dm_cheetah_turbo_enn_fit.toml",
    "configs/bo/atari/exp_atari_pong_frozen_turbo_enn_p.toml",
)

_TARGET_RL_TOMLS = (
    "configs/rl/gymnasium/bw/ppo_puffer_cuda.toml",
    "configs/rl/gymnasium/bw/ppo_puffer_macos.toml",
    "configs/rl/atari/ppo_pong_puffer_cuda.toml",
    "configs/rl/atari/ppo_pong_puffer_macos.toml",
)

_ALL_BO_TOMLS = tuple(sorted((_REPO_ROOT / "configs" / "bo").rglob("*.toml")))
_ALL_RL_TOMLS = tuple(sorted((_REPO_ROOT / "configs" / "rl").rglob("*.toml")))


def _install_stub_experiment_sampler(monkeypatch):
    calls = []

    class _StubExperimentConfig:
        @classmethod
        def from_dict(cls, d):
            return SimpleNamespace(**d)

    scan_local_sentinel = object()

    def _sampler(config, distributor_fn):
        calls.append((config, distributor_fn))

    mod = types.ModuleType("experiments.experiment_sampler")
    mod.ExperimentConfig = _StubExperimentConfig
    mod.sampler = _sampler
    mod.scan_local = scan_local_sentinel
    monkeypatch.setitem(sys.modules, "experiments.experiment_sampler", mod)
    return calls, scan_local_sentinel


@pytest.mark.parametrize("rel_path", _TARGET_BO_TOMLS)
def test_bo_target_tomls_cli_smoke(monkeypatch, rel_path):
    calls, scan_local_sentinel = _install_stub_experiment_sampler(monkeypatch)
    result = CliRunner().invoke(bo_cli, ["local", str(_REPO_ROOT / rel_path)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    cfg, distributor_fn = calls[0]
    assert isinstance(cfg.env_tag, str) and cfg.env_tag
    assert isinstance(cfg.opt_name, str) and cfg.opt_name
    assert distributor_fn is scan_local_sentinel


@pytest.mark.parametrize("rel_path", _TARGET_RL_TOMLS)
def test_rl_target_tomls_runner_smoke(monkeypatch, rel_path):
    calls = []

    class _StubAlgoConfig:
        def __init__(self, payload):
            self.payload = payload
            self.seed = int(payload.get("seed", 0))
            self.exp_dir = str(payload.get("exp_dir", "_tmp/stub"))

        @classmethod
        def from_dict(cls, d):
            return cls(dict(d))

    def _train_fn(config):
        calls.append(config)
        return {"ok": True}

    algo = SimpleNamespace(config_cls=_StubAlgoConfig, train_fn=_train_fn)
    monkeypatch.setattr(rl_builtins, "register_all", lambda: None)
    monkeypatch.setattr(rl_runner, "get_algo", lambda _name, backend=None: algo)
    rl_runner.main(["--config", str(_REPO_ROOT / rel_path)])
    assert len(calls) == 1
    assert isinstance(calls[0].seed, int)
    assert isinstance(calls[0].exp_dir, str) and calls[0].exp_dir


@pytest.mark.parametrize(
    "toml_path",
    _ALL_BO_TOMLS,
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_all_bo_tomls_parse(toml_path):
    cfg = load_experiment_config(config_toml_path=str(toml_path))
    assert isinstance(cfg.env_tag, str) and cfg.env_tag
    assert isinstance(cfg.exp_dir, str) and cfg.exp_dir
    assert isinstance(cfg.opt_name, str) and cfg.opt_name


@pytest.mark.parametrize(
    "toml_path",
    _ALL_RL_TOMLS,
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_all_rl_tomls_parse_for_runner(toml_path):
    cfg = load_toml(str(toml_path))
    if cfg == {} and toml_path.stat().st_size == 0:
        pytest.skip("Placeholder TOML is intentionally empty.")
    algo_name, _backend, algo_cfg = rl_runner._extract_algo_cfg(cfg)
    seeds, workers = rl_runner._extract_run_cfg(cfg)
    assert isinstance(algo_name, str) and algo_name
    assert isinstance(algo_cfg, dict)
    assert isinstance(seeds, list)
    assert workers >= 1
