from __future__ import annotations

import importlib
from types import SimpleNamespace

from click.testing import CliRunner


def _fake_im(erun, name):
    if name == "ops.exp_uhd_parse":
        from ops.uhd_config import UHDConfig

        return SimpleNamespace(
            _load_toml_config=lambda p: {"uhd": {"env_tag": "mnist", "num_rounds": 1, "optimizer": "mezo"}},
            _validate_required=lambda c: None,
            _parse_cfg=lambda c: UHDConfig(
                env_tag="mnist",
                num_rounds=1,
                lr=0.01,
                sigma=0.001,
                optimizer="mezo",
                num_dim_target=2,
                num_module_target=1,
                policy_tag="pure-function",
            ),
        )
    if name == "tomllib":
        import tomllib as tl

        return tl
    if name == "ops.modal_uhd":
        return SimpleNamespace(run=lambda *a, **k: "modal-log")
    if name == "ops.exp_uhd_run":
        return erun
    raise AssertionError(name)


def test_kiss_tidy_b_ops_cli_modal_uhd(monkeypatch, tmp_path):
    ecli = importlib.import_module("ops.exp_uhd_cli")
    erun = importlib.import_module("ops.exp_uhd_run")
    r = CliRunner()
    tom = tmp_path / "u.toml"
    tom.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')

    monkeypatch.setattr("common.im.im", lambda name: _fake_im(erun, name))
    monkeypatch.setattr(erun, "uhd_config_toml_to_modal_log", lambda *a, **k: "ok")
    assert r.invoke(ecli.cli, ["modal", str(tom)]).exit_code == 0
    assert "ok" in erun.uhd_config_toml_to_modal_log(
        str(tom),
        "A100",
        exp_uhd_parse=_fake_im(erun, "ops.exp_uhd_parse"),
        tomllib=__import__("tomllib"),
        modal_run=lambda *a, **k: "ok",
    )

    modal_uhd = importlib.import_module("ops.modal_uhd")
    modal_uhd_runner_impl = importlib.import_module("ops.modal_uhd_runner_impl")
    monkeypatch.setattr(modal_uhd_runner_impl, "run", lambda *a, **k: "MR")
    assert modal_uhd.run("mnist", 1, 0.01, 2, 1, policy_tag="pure-function", gpu="cpu") == "MR"

    exp_uhd_full = importlib.import_module("ops.exp_uhd_full")
    monkeypatch.setattr(erun, "uhd_config_toml_to_modal_log", lambda *a, **k: "full-ok")
    exp_uhd_full.modal_cmd.callback(str(tom), None, "A100")

    monkeypatch.setattr("ops.uhd_batch_cli._load_toml", lambda p: {"env_tag": "mnist", "num_rounds": 1})
    monkeypatch.setattr("ops.uhd_batch_cli._batch_modal", lambda *a, **k: None)
    monkeypatch.setattr("ops.uhd_batch_cli._collect", lambda *a, **k: None)
    ubc = importlib.import_module("ops.uhd_batch_cli")
    t2 = tmp_path / "b.toml"
    t2.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')
    assert r.invoke(ubc.cli, ["submit", "--config", str(t2), "--num-reps", "1"]).exit_code == 0
    assert r.invoke(ubc.cli, ["collect"]).exit_code == 0
