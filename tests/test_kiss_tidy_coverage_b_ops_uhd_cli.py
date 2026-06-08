from __future__ import annotations

import importlib
import subprocess
from types import SimpleNamespace
from unittest.mock import patch


def test_kiss_tidy_b_ops_cli_local_uhd(monkeypatch, tmp_path):
    erun = importlib.import_module("ops.exp_uhd_run")
    register_local_commands = importlib.import_module("ops.enn_incremental_batches_local").register_local_commands
    modal_cli_common = importlib.import_module("ops.modal_cli_common")

    assert callable(register_local_commands)

    def _run(*a, **k):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr("ops.modal_batches.sys.exit", lambda *a, **k: None)
    monkeypatch.setattr("ops.synthetic_sine_benchmark_batches.sys.exit", lambda *a, **k: None)
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
    modal_cli_common.run_modal(["deploy", "--help"], "t", run=_run)
    modal_cli_common.run_modal_entrypoint("experiments.modal_enn_incremental_batches_impl", "t", "status", run=_run)
    modal_cli_common.collect_to_output_dir(
        "experiments.modal_enn_incremental_batches_impl",
        "t",
        str(tmp_path),
        run=_run,
    )
    modal_cli_common.stop_app_and_delete_dicts(app_name="yubo-enn-incremental-t", dict_names=["d1"], run=_run)

    tom = tmp_path / "u.toml"
    tom.write_text(
        '[uhd]\nenv_tag = "mnist"\npolicy_tag = "pure-function"\nnum_rounds = 1\noptimizer = "mezo"\n'
        'lr = 0.01\nperturb = "dim:2"\n'
        "problem_seed = 0\nnoise_seed_0 = 0\nbatch_size = 4\nlog_interval = 1\n"
        "accuracy_interval = 1000\n"
    )

    def _dummy_loop(*_a, **_k):
        return SimpleNamespace(run=lambda: None)

    parsed = SimpleNamespace(
        optimizer="mezo",
        env_tag="mnist",
        num_rounds=1,
        lr=0.01,
        sigma=0.001,
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
        be=None,
    )
    with patch("ops.uhd_setup_make_loop.make_loop", _dummy_loop):
        erun.run_local_from_toml(str(tom))
        erun.run_parsed_uhd_local(parsed)

    monkeypatch.setattr("ops.uhd_setup_bszo.run_bszo_loop", lambda *a, **k: None)
    with patch("ops.uhd_setup_make_loop.make_loop", _dummy_loop):
        erun.run_parsed_uhd_local(SimpleNamespace(**{**parsed.__dict__, "optimizer": "simple", "num_module_target": None, "lr": 0.001}))
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
