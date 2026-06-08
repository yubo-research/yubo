from __future__ import annotations

from types import SimpleNamespace

import torch

from ops.exp_uhd import modal_cmd
from rl.runner import main


def test_kiss_cov_direct_exp_uhd_modal_and_runner_main(monkeypatch, tmp_path):
    import common.config_toml as ct
    import rl.runner as runner
    import rl.runner_helpers as rh

    monkeypatch.setattr("ops.modal_uhd.run", lambda *args, **kwargs: "ok")
    toml = tmp_path / "cfg.toml"
    toml.write_text('[uhd]\nenv_tag="f:sphere-2d"\nnum_rounds=1\n')
    modal_cmd(str(toml), (), None, "A100")

    cfg_path = tmp_path / "rl.toml"
    # Use valid policy_tag 'mlp-16-8'
    cfg_path.write_text('[rl]\nalgo="dummy"\n[rl.dummy]\nenv_tag="pend"\npolicy_tag="mlp-16-8"\nseed=7\nexp_dir="tmp"\n')

    monkeypatch.setattr(rh, "split_config_and_args", lambda argv: (str(cfg_path), argv[1:]))
    monkeypatch.setattr(
        rh,
        "parse_runtime_args",
        lambda rest: SimpleNamespace(workers=1, workers_cli_set=False, cleaned=rest),
    )
    monkeypatch.setattr(ct, "parse_set_args", lambda cleaned: {})
    monkeypatch.setattr(
        ct,
        "load_toml",
        lambda _path: {
            "rl": {
                "algo": "dummy",
                "dummy": {
                    "env_tag": "pend",
                    "policy_tag": "mlp-16-8",
                    "seed": 7,
                    "exp_dir": "tmp",
                },
            }
        },
    )
    monkeypatch.setattr(ct, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(runner, "_extract_run_cfg", lambda cfg: ([], 1))

    # Use a real config class that matches the expected structure
    from rl.torchrl.sac.config import SACConfig as _Cfg

    def _train_fn(_cfg):
        return {"ok": True}

    monkeypatch.setattr(
        "rl.registry.get_algo",
        lambda _algo_name: SimpleNamespace(config_cls=_Cfg, train_fn=_train_fn),
    )
    monkeypatch.setattr("rl.builtins.register_all", lambda: None)

    main(["local", "--config", "config.toml"])


def test_kiss_cov_direct_torchrl_replay_buffer(monkeypatch, tmp_path):
    from rl.core.replay import make_replay_buffer

    rb = make_replay_buffer(obs_shape=(4,), act_dim=2, capacity=10, backend="numpy")
    assert rb is not None


def test_kiss_cov_direct_torchrl_sac_models(monkeypatch):
    from rl.core.runtime import ObsScaler
    from rl.torchrl.offpolicy.models import ActorNet, QNet

    obs_scaler = ObsScaler(None, None)

    backbone = torch.nn.Linear(4, 8)
    head = torch.nn.Linear(8, 2)

    # ActorNet signature: (backbone, head, obs_scaler, act_dim)
    actor = ActorNet(backbone, head, obs_scaler, act_dim=2)
    assert actor is not None

    # QNet signature: (backbone, head, obs_scaler)
    qnet = QNet(backbone, head, obs_scaler)
    assert qnet is not None
