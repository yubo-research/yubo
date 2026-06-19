from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script


def _decoded_modal_build_command(command: str) -> str:
    payloads = _decoded_modal_build_payloads(command)
    assert payloads
    return payloads[0]


def _decoded_modal_build_payloads(command: str) -> list[str]:
    import base64
    import re

    payloads = [
        base64.b64decode(match.group("payload")).decode("utf-8") for match in re.finditer(r"printf %s '?(?P<payload>[A-Za-z0-9+/=]+)'? \| base64 -d", command)
    ]
    return payloads


def test_modal_pixi_base_image_uses_cacheable_layers() -> None:
    import inspect

    import ops.modal_pixi_base_image as base

    source = inspect.getsource(base.mk_pixi_base_image)
    isaac_source = inspect.getsource(base._isaaclab_install_commands)
    assert "add_local_dir" not in source
    assert "micromamba" not in source
    assert "debian_slim" in source
    assert "_isaaclab_install_commands()" in source
    assert "_pixi_install_env_command(ISAACLAB_PIXI_ENV)" in isaac_source
    assert source.index("_pixi_install_env_command(PRIMARY_PIXI_ENV)") >= 0
    assert base.PIXI_MANIFEST_PATH.endswith("/pixi.toml")
    assert base.PIXI_LOCK_PATH.endswith("/pixi.lock")
    assert base.PIXI_CHECK_PATH.endswith("/admin/check_pixi_env.py")
    assert base.ENN_PATCH_PATH.endswith("/admin/patch_enn_failure_tolerance_dim.py")
    assert '"admin" / "check_pixi_env.py"' in source
    assert '"admin" / "patch_enn_failure_tolerance_dim.py"' in source

    info_command = _decoded_modal_build_command(base._pixi_info_command())
    assert "pixi info" in info_command
    assert base.PIXI_MANIFEST_PATH in info_command

    install_command = _decoded_modal_build_command(base._pixi_install_env_command(base.PRIMARY_PIXI_ENV))
    assert "pixi install" in install_command
    assert "--locked -e yubo" in install_command

    setup_command = _decoded_modal_build_command(base._pixi_task_command(base.PRIMARY_PIXI_ENV, "setup"))
    assert "pixi run" in setup_command
    assert "--locked -e yubo setup" in setup_command

    check_command = _decoded_modal_build_command(base._primary_check_command())
    assert "LD_LIBRARY_PATH" in check_command
    assert "/opt/yubo-pixi/.pixi/envs/yubo/lib" in check_command
    assert "pixi run" in check_command
    assert "--locked -e yubo check" in check_command

    bootstrap = base.isaaclab_bootstrap_command()
    assert "already installed, skipping" in bootstrap
    assert ".isaaclab_bootstrap_ok" in bootstrap
    bootstrap_payload = "\n".join(_decoded_modal_build_payloads(bootstrap))
    assert "pixi install" in bootstrap_payload
    assert "-e isaaclab setup" in bootstrap_payload

    force_payload = "\n".join(_decoded_modal_build_payloads(base.isaaclab_bootstrap_command(force=True)))
    assert "pixi install" in force_payload
    assert "-e isaaclab" in force_payload


def test_nvidia_vulkan_icd_script_prefers_real_icd() -> None:
    script = nvidia_vulkan_icd_script()
    assert "/usr/share/vulkan/icd.d/nvidia_icd.json" in script
    assert "/etc/vulkan/icd.d/nvidia_icd.json" in script
    assert "unset VK_ICD_FILENAMES" in script


def test_modal_isaac_render_probe_helpers_importable() -> None:
    pytest = __import__("pytest")
    try:
        import ops.modal_isaac_render_probe as probe
    except ImportError as exc:
        if exc.name == "modal":
            pytest.skip("modal package is not installed")
        raise

    assert "vulkaninfo --summary" in probe._probe_command()
    assert "isaacsim[all,extscache]==6.0.0.0" in probe._isaacsim_smoke_command()
    assert "DEFAULT_LAUNCHER_CONFIG" in probe._official_inspect_command()
    assert "/dev/nvidia-caps" in probe._official_device_caps_command()
    assert "vulkaninfo --summary" in probe._official_probe_command()
    assert "OFFICIAL_ISAACSIM_SIMULATION_APP_OK" in probe._official_isaacsim_smoke_command()
    assert "OFFICIAL_ISAACSIM_MINIMAL_OK" in probe._official_isaacsim_minimal_command()
    assert "disable_viewport_updates" in probe._official_isaacsim_minimal_command(disable_viewport_updates=True)
    assert "--exec /tmp/yubo_kit_smoke.py" in probe._official_kit_smoke_command()
    assert "OFFICIAL_KIT_SMOKE_LIGHT_OK" in probe._official_kit_smoke_command(disable_heavy_extensions=True)
    assert "OFFICIAL_ISAACLAB_ROLLOUT_RETURN" in probe._official_isaaclab_rollout_video_command()
    assert "OFFICIAL_KIT_RENDER_CAPTURE_OK" in probe._official_kit_render_capture_command()
    assert "OFFICIAL_KIT_BARE_RENDER_CAPTURE_OK" in probe._official_kit_bare_render_capture_command()
    assert "OFFICIAL_KIT_STORM_RENDER_CAPTURE_OK" in probe._official_kit_storm_render_capture_command()


def test_modal_pixi_setup_runtime_command_script() -> None:
    pytest = __import__("pytest")
    try:
        import ops.modal_pixi_setup as setup
    except ImportError as exc:
        if exc.name == "modal":
            pytest.skip("modal package is not installed")
        raise

    assert setup._setup_complete_command() == "echo modal pixi setup ok"
    command_script = setup._runtime_command_script("echo mujoco_playground:CheetahRun", setup.PRIMARY_PIXI_ENV)
    assert "VK_ICD_FILENAMES" in command_script
    assert "PIXI_HOME=/opt/pixi" in command_script
    assert "YUBO_PIXI_PREFIX" in command_script
    assert "echo mujoco_playground:CheetahRun" in command_script
    assert "micromamba" not in command_script
    pytest_command = setup._pytest_command(setup.PRIMARY_PIXI_ENV, "--no-testmon -q tests/test_hs_llm.py")
    assert "env JAX_PLATFORMS=cuda,cpu python -m pytest --no-testmon -q tests/test_hs_llm.py" in pytest_command
    pytest_cpu_platforms = setup._resolve_pytest_jax_platforms(pytest_cpu=True, jax_platforms="")
    assert pytest_cpu_platforms == "cpu"
    assert setup._resolve_pytest_jax_platforms(pytest_cpu=False, jax_platforms="cpu") == "cpu"
    assert setup._remote_runner(pytest=True, with_export=False, pytest_cpu=False) is setup.run_pixi_command
    assert setup._remote_runner(pytest=True, with_export=False, pytest_cpu=True) is setup.run_pixi_command_cpu


def test_resolve_pixi_env_isaac_jax_sim_uses_primary_env() -> None:
    pytest = __import__("pytest")
    try:
        import ops.modal_pixi_setup as setup
    except ImportError as exc:
        if exc.name == "modal":
            pytest.skip("modal package is not installed")
        raise

    assert setup._resolve_pixi_env("auto", "configs/bo/isaaclab/g1_flat_eggroll_jax_smoke.toml") == setup.PRIMARY_PIXI_ENV
    assert setup._resolve_pixi_env("auto", "configs/bo/isaaclab/g1_flat_eggroll_dev.toml") == setup.ISAACLAB_PIXI_ENV
    prefix = setup._isaaclab_runtime_prefix("configs/bo/isaaclab/g1_flat_eggroll_jax_smoke.toml")
    assert "PYTHONPATH=" in prefix
    assert "/src/IsaacLab/source/isaaclab" in prefix
    exp_cmd = setup._experiment_command(
        setup.PRIMARY_PIXI_ENV,
        "local",
        "configs/bo/isaaclab/g1_flat_eggroll_jax_smoke.toml",
    )
    assert "env PYTHONPATH=" in exp_cmd
    assert "/src/IsaacLab/source/isaaclab" in exp_cmd
    assert setup._isaaclab_runtime_prefix("configs/bo/isaaclab/g1_flat_eggroll_dev.toml") == ""
