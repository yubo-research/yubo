from __future__ import annotations

import subprocess
from types import SimpleNamespace

from click.testing import CliRunner

import ops.synthetic_sine_benchmark_batches as ssbo
from ops.synthetic_sine_benchmark_batches import (
    cli,
    collect,
    deploy,
    local_single,
    status,
    stop,
    submit,
)


def test_kiss_tidy_b_ops_synthetic_sine_batches_cli(monkeypatch, tmp_path):
    def _run(*a, **k):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(ssbo.sys, "exit", lambda *a, **k: None)
    import analysis.fitting_time.evaluate as evm

    monkeypatch.setattr(evm, "benchmark_single_surrogate_with_data", lambda **k: (1.0, 2.0, 3.0))
    r = CliRunner()
    for cmd, args in (
        (cli, ["deploy", "t"]),
        (deploy, ["deploy", "t"]),
        (submit, ["submit", "t", "example_sphere_n12_d2_seed0"]),
        (collect, ["collect", "t"]),
        (status, ["status", "t"]),
        (stop, ["stop", "t"]),
    ):
        assert r.invoke(cmd, args).exit_code == 0
    deploy("t")
    submit("t", "example_sphere_n12_d2_seed0")
    assert callable(local_single)
    outd = tmp_path / "ssb"
    assert (
        r.invoke(
            cli,
            [
                "local-single",
                "8",
                "sphere",
                "0",
                "gp",
                "--output-dir",
                str(outd),
                "--num-reps",
                "1",
            ],
        ).exit_code
        == 0
    )
