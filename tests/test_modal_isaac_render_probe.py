from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script


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


def test_modal_hyperscalees_setup_exposes_isaac_preflight() -> None:
    pytest = __import__("pytest")
    try:
        import ops.modal_hyperscalees_setup as setup
    except ImportError as exc:
        if exc.name == "modal":
            pytest.skip("modal package is not installed")
        raise

    assert "problems.isaaclab_env_adapters" in setup._isaaclab_preflight_command()
    assert "isaacsim" in setup._isaaclab_preflight_command()
    assert "isaaclab_default_launcher_kwargs" in setup._isaaclab_preflight_command()
