from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script


def _decoded_modal_build_command(command: str) -> str:
    import base64
    import re

    match = re.search(r"printf %s '?(?P<payload>[A-Za-z0-9+/=]+)'? \| base64 -d", command)
    assert match is not None
    return base64.b64decode(match.group("payload")).decode("utf-8")


def test_modal_hyperscalees_pixi_base_image_uses_cacheable_layers() -> None:
    import inspect

    import ops.modal_hyperscalees_pixi_base_image as base

    source = inspect.getsource(base.mk_hyperscalees_pixi_base_image)
    assert "add_local_dir" not in source
    assert "micromamba" not in source
    assert "debian_slim" in source
    assert f"_pixi_install_env_command({base.ISAACLAB_PIXI_ENV})" not in source
    assert source.index("_pixi_install_env_command(HYPERSCALEES_PIXI_ENV)") >= 0
    assert base.PIXI_MANIFEST_PATH.endswith("/pixi.toml")
    assert base.PIXI_LOCK_PATH.endswith("/pixi.lock")

    info_command = _decoded_modal_build_command(base._pixi_info_command())
    assert "pixi info" in info_command
    assert base.PIXI_MANIFEST_PATH in info_command

    install_command = _decoded_modal_build_command(base._pixi_install_env_command(base.HYPERSCALEES_PIXI_ENV))
    assert "pixi install" in install_command
    assert "--locked -e hyperscalees" in install_command

    setup_command = _decoded_modal_build_command(base._pixi_task_command(base.HYPERSCALEES_PIXI_ENV, "setup"))
    assert "pixi run" in setup_command
    assert "--locked -e hyperscalees setup" in setup_command

    check_command = _decoded_modal_build_command(base._hyperscalees_check_command())
    assert "LD_LIBRARY_PATH" in check_command
    assert "/opt/yubo-pixi/.pixi/envs/hyperscalees/lib" in check_command
    assert "EpistemicNearestNeighbors" in check_command
    assert "pixi run" not in check_command

    bootstrap = base.isaaclab_bootstrap_command()
    bootstrap_install = _decoded_modal_build_command(bootstrap.split("; ")[0])
    assert "pixi install" in bootstrap_install
    assert "-e isaaclab" in bootstrap_install
    bootstrap_setup = _decoded_modal_build_command(bootstrap.split("; ")[1])
    assert "pixi run" in bootstrap_setup
    assert "-e isaaclab install" in bootstrap_setup


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


def test_modal_hyperscalees_pixi_setup_exposes_isaac_preflight() -> None:
    pytest = __import__("pytest")
    try:
        import ops.modal_hyperscalees_pixi_setup as setup
    except ImportError as exc:
        if exc.name == "modal":
            pytest.skip("modal package is not installed")
        raise

    assert "problems.isaaclab_env_adapters" in setup._isaaclab_preflight_command()
    assert "isaacsim" in setup._isaaclab_preflight_command()
    assert "isaaclab_default_launcher_kwargs" in setup._isaaclab_preflight_command()
    command_script = setup._runtime_command_script("echo mujoco_playground:CheetahRun", setup.HYPERSCALEES_PIXI_ENV)
    assert "VK_ICD_FILENAMES" in command_script
    assert "PIXI_HOME=/opt/pixi" in command_script
    assert "YUBO_PIXI_PREFIX" in command_script
    assert "echo mujoco_playground:CheetahRun" in command_script
    assert "micromamba" not in command_script
