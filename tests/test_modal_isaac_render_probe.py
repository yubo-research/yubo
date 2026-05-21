from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script


def _decoded_modal_build_command(command: str) -> str:
    import base64
    import re

    match = re.search(r"printf %s '?(?P<payload>[A-Za-z0-9+/=]+)'? \| base64 -d", command)
    assert match is not None
    return base64.b64decode(match.group("payload")).decode("utf-8")


def test_modal_hyperscalees_base_image_uses_cacheable_layers() -> None:
    import inspect

    import ops.modal_hyperscalees_base_image as base

    source = inspect.getsource(base.mk_hyperscalees_base_image)
    assert "add_local_dir" not in source
    assert "setup-hyperscalees.sh" not in source
    assert "run_hyperscalees_install" not in source
    assert "requirements-isaaclab.txt" in source
    assert source.index("conda_isaaclab_yml") < source.index("conda_hyperscalees_yml")
    assert source.index("_validate_isaaclab_runtime_command") < source.index("_install_python_requirements_command")

    conda_command = base._create_conda_env_command("channels: []\ndependencies: []\n")
    assert "\n" not in conda_command
    decoded_conda_command = _decoded_modal_build_command(conda_command)
    assert "cat > /tmp/yubo-conda-hyperscalees.yml" in decoded_conda_command
    assert "micromamba env create" in decoded_conda_command
    assert "/root/admin" not in decoded_conda_command

    source_command = _decoded_modal_build_command(base._install_source_extras_command())
    assert "HyperscaleES.git" in source_command
    assert "VecchiaBO.git" in source_command
    assert "TORCH_CUDA_ARCH_LIST" in source_command
    assert "setup-hyperscalees.sh" not in source_command

    final_command = _decoded_modal_build_command(base._finalize_runtime_compat_command())
    assert "libfaiss=1.10.0=cpu_openblas*" in final_command
    assert "faiss=1.10.0=cpu_openblas_py312*" in final_command
    assert "setuptools>=77.0.3,<81.0.0" in final_command
    assert "pkg_resources" in final_command
    assert "numba==0.61.2" in final_command
    assert "llvmlite==0.44.0" in final_command
    assert "MODAL_HYPERSCALEES_FINAL_OK" in final_command


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
    command_script = setup._runtime_command_script("echo brax:ant")
    assert "VK_ICD_FILENAMES" in command_script
    assert "LD_LIBRARY_PATH=/opt/conda/envs/yubo-hyperscalees/lib" in command_script
    assert "echo brax:ant" in command_script
    assert "yubo_brax_compat_check.py" not in command_script
