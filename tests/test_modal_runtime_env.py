from __future__ import annotations

from ops import modal_runtime_env as runtime_env


def test_modal_runtime_env_detects_local_config_with_overrides(tmp_path):
    cfg = tmp_path / "g1.toml"
    cfg.write_text('[experiment]\nenv_tag = "isaaclab:Isaac-Velocity-Flat-G1-v0"\n', encoding="utf-8")

    args = ["local", "--opt", "num_rounds=2", str(cfg)]

    assert runtime_env.local_experiment_config_arg(args) == str(cfg)
    assert runtime_env.experiment_env_tag(cfg) == "isaaclab:Isaac-Velocity-Flat-G1-v0"
    assert runtime_env.target_env_for_config(cfg) == runtime_env.ISAACLAB_ENV_NAME


def test_modal_runtime_env_ignores_non_isaaclab_config(tmp_path):
    cfg = tmp_path / "cheetah.toml"
    cfg.write_text('[experiment]\nenv_tag = "cheetah"\n', encoding="utf-8")

    assert runtime_env.target_env_for_config(cfg) is None


def test_modal_runtime_env_filters_stale_conda_library_paths():
    existing = "/opt/conda/envs/yubo-hyperscalees/lib:/usr/local/cuda/lib64:/opt/conda/envs/yubo-isaaclab/lib"

    assert runtime_env.filtered_ld_library_path(runtime_env.ISAACLAB_ENV_NAME, existing) == "/opt/conda/envs/yubo-isaaclab/lib:/usr/local/cuda/lib64"


def test_modal_runtime_env_reexec_command_and_environ():
    env = {
        "LD_LIBRARY_PATH": "/opt/conda/envs/yubo-hyperscalees/lib:/usr/local/cuda/lib64",
        "PYTHONPATH": "/root",
    }

    cmd = runtime_env.reexec_command(
        target_env=runtime_env.ISAACLAB_ENV_NAME,
        script_path="/root/ops/experiment.py",
        args=["local", "configs/bo/isaaclab/g1_flat_ppo_dev.toml"],
        environ=env,
    )
    routed_env = runtime_env.reexec_environ(runtime_env.ISAACLAB_ENV_NAME, env)

    assert cmd[:4] == ["micromamba", "run", "-n", runtime_env.ISAACLAB_ENV_NAME]
    assert "LD_LIBRARY_PATH=/opt/conda/envs/yubo-isaaclab/lib:/usr/local/cuda/lib64" in cmd
    assert routed_env["LD_LIBRARY_PATH"] == "/opt/conda/envs/yubo-isaaclab/lib:/usr/local/cuda/lib64"
    assert routed_env["PYTHONPATH"] == "/root"
