from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

import ops.exp_uhd_cli as ecli
import ops.exp_uhd_run as erun
import ops.modal_batches as omb
import ops.synthetic_sine_benchmark_batches as ssbo
import ops.uhd_batch_cli as ubc
from ops.enn_incremental_batches_local import register_local_commands
from ops.exp_uhd_cli import modal_cmd
from ops.exp_uhd_full import modal_cmd as exp_uhd_full_modal_cmd
from ops.modal_cli_common import (
    collect_to_output_dir,
    run_modal,
    run_modal_entrypoint,
    stop_app_and_delete_dicts,
)
from ops.modal_uhd import run
from ops.modal_uhd_runner_impl import run as modal_uhd_runner_impl_run
from ops.uhd_batch_cli import collect_cmd
from ops.uhd_batch_cli import modal_cmd as uhd_batch_modal_cmd


def _runner():
    return CliRunner()


def test_kiss_tidy_b_ops_cli_batches_uhd(monkeypatch, tmp_path):
    assert callable(register_local_commands)

    def _run(*a, **k):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _run)
    for mod in (omb, ssbo):
        monkeypatch.setattr(mod.sys, "exit", lambda *a, **k: None)
    from modal_timing_sweep_test_support import FakeResultsDict

    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Dict.from_name",
        lambda *a, **k: FakeResultsDict(),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Function.from_name",
        lambda *a, **k: SimpleNamespace(spawn=lambda *sa, **sk: None, spawn_map=lambda *sa, **sk: None),
    )
    monkeypatch.setattr("ops.modal_cli_common.sys.exit", lambda code=0: None)
    run_modal(["deploy", "--help"], "t", run=_run)
    run_modal_entrypoint("experiments.modal_enn_incremental_batches_impl", "t", "status", run=_run)
    collect_to_output_dir(
        "experiments.modal_enn_incremental_batches_impl",
        "t",
        str(tmp_path),
        run=_run,
    )
    stop_app_and_delete_dicts(app_name="yubo-enn-incremental-t", dict_names=["d1"], run=_run)
    r = _runner()

    tom = tmp_path / "u.toml"
    tom.write_text(
        '[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\noptimizer = "mezo"\n'
        'lr = 0.01\nnum_dim_target = 2\nnum_module_target = 1\npolicy_tag = "pure-function"\n'
        "problem_seed = 0\nnoise_seed_0 = 0\nbatch_size = 4\nlog_interval = 1\n"
        "accuracy_interval = 1000\n"
    )

    with patch("optimizer.uhd_loop.UHDLoop", lambda *a, **k: SimpleNamespace(run=lambda: None)):
        erun.run_parsed_uhd_local(
            SimpleNamespace(
                optimizer="mezo",
                env_tag="mnist",
                num_rounds=1,
                lr=0.01,
                num_dim_target=2,
                num_module_target=1,
                policy_tag="pure-function",
                problem_seed=0,
                noise_seed_0=0,
                batch_size=4,
                log_interval=1,
                accuracy_interval=1000,
                target_accuracy=None,
                early_reject=None,
                enn=None,
            )
        )

    monkeypatch.setattr("ops.uhd_setup_simple_gym.run_simple_loop", lambda *a, **k: None)
    monkeypatch.setattr("ops.uhd_setup_bszo.run_bszo_loop", lambda *a, **k: None)
    with patch("optimizer.uhd_loop.UHDLoop", lambda *a, **k: SimpleNamespace(run=lambda: None)):
        erun.run_parsed_uhd_local(
            SimpleNamespace(
                optimizer="simple",
                env_tag="mnist",
                num_rounds=1,
                num_dim_target=2,
                policy_tag="pure-function",
                problem_seed=0,
                noise_seed_0=0,
                batch_size=4,
                log_interval=1,
                accuracy_interval=1000,
                target_accuracy=None,
                be=None,
            )
        )
    erun.run_parsed_uhd_local(
        SimpleNamespace(
            optimizer="bszo",
            env_tag="mnist",
            num_rounds=1,
            lr=0.01,
            policy_tag="pure-function",
            problem_seed=0,
            noise_seed_0=0,
            batch_size=4,
            log_interval=1,
            accuracy_interval=1000,
            target_accuracy=None,
            bszo_k=2,
            bszo_epsilon=1e-4,
            bszo_sigma_p_sq=1.0,
            bszo_sigma_e_sq=1.0,
            bszo_alpha=0.1,
        )
    )

    def _fake_im(name):
        if name == "ops.exp_uhd_parse":
            return SimpleNamespace(
                _load_toml_config=lambda p: {
                    "uhd": {
                        "env_tag": "mnist",
                        "num_rounds": 1,
                        "optimizer": "mezo",
                        "lr": 0.01,
                        "num_dim_target": 2,
                        "num_module_target": 1,
                        "policy_tag": "pure-function",
                        "problem_seed": 0,
                        "noise_seed_0": 0,
                        "batch_size": 4,
                        "log_interval": 1,
                        "accuracy_interval": 1000,
                    }
                },
                _validate_required=lambda c: None,
                _parse_cfg=lambda c: SimpleNamespace(
                    env_tag="mnist",
                    num_rounds=1,
                    lr=0.01,
                    num_dim_target=2,
                    num_module_target=1,
                    policy_tag="pure-function",
                    problem_seed=0,
                    noise_seed_0=0,
                    log_interval=1,
                    accuracy_interval=1000,
                    target_accuracy=None,
                    early_reject=None,
                    enn=None,
                ),
            )
        if name == "tomllib":
            import tomllib as tl

            return tl
        if name == "ops.modal_uhd":
            return SimpleNamespace(run=lambda *a, **k: "modal-log")
        raise AssertionError(name)

    monkeypatch.setattr("common.im.im", _fake_im)
    assert r.invoke(ecli.cli, ["modal", str(tom)]).exit_code == 0
    assert "ok" in erun.uhd_config_toml_to_modal_log(
        str(tom),
        "A100",
        exp_uhd_parse=_fake_im("ops.exp_uhd_parse"),
        tomllib=__import__("tomllib"),
        modal_run=lambda *a, **k: "ok",
    )

    monkeypatch.setattr("ops.modal_uhd_runner_impl.run", lambda *a, **k: "MR")
    assert run("mnist", 1, 0.01, 2, 1, gpu="cpu", problem_seed=0, noise_seed_0=0) == "MR"
    assert callable(modal_uhd_runner_impl_run)

    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: "full-ok")
    exp_uhd_full_modal_cmd(str(tom), None, "A100")
    assert callable(modal_cmd)

    monkeypatch.setattr(
        "ops.uhd_batch_cli._load_toml",
        lambda p: {"env_tag": "mnist", "num_rounds": 1, "optimizer": "mezo"},
    )
    monkeypatch.setattr("ops.uhd_batch_cli._batch_modal", lambda *a, **k: None)
    monkeypatch.setattr("ops.uhd_batch_cli._collect", lambda *a, **k: None)
    t2 = tmp_path / "b.toml"
    t2.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')
    assert r.invoke(ubc.cli, ["modal", str(t2), "--num-reps", "1"]).exit_code == 0
    assert callable(uhd_batch_modal_cmd)
    assert callable(collect_cmd)
    assert r.invoke(ubc.cli, ["collect"]).exit_code == 0
