from pathlib import Path

from experiments import experiment_toml


def test_run_workers_selects_parallel(tmp_path, monkeypatch):
    config_path = Path(tmp_path) / "config.toml"
    config_path.write_text(
        """
[experiment]
exp_dir = "_tmp/exp"
env_tag = "f:ackley-2d"
opt_name = "ucb"
num_arms = 2
num_rounds = 1
num_reps = 1
run_workers = 2
"""
    )

    calls = {"parallel": False, "local": False}

    def fake_parallel(run_configs, max_total_seconds=None, *, max_workers):
        calls["parallel"] = True
        assert max_workers == 2

    def fake_local(run_configs, max_total_seconds=None):
        calls["local"] = True

    def fake_sampler(config, distributor_fn):
        distributor_fn([], max_total_seconds=None)

    monkeypatch.setattr(experiment_toml, "scan_parallel", fake_parallel)
    monkeypatch.setattr(experiment_toml, "scan_local", fake_local)
    monkeypatch.setattr(experiment_toml, "sampler", fake_sampler)

    experiment_toml.main(["--config", str(config_path)])

    assert calls["parallel"] is True
    assert calls["local"] is False
