from __future__ import annotations

from ops.modal_isaac_render_commands import _vk_env_script

_HEAVY_EXTENSION_DISABLE_ARGS = (
    "--disable omni.replicator.core",
    "--disable omni.replicator.nv",
    "--disable omni.replicator.replicator_yaml",
    "--disable isaacsim.sensors.rtx",
    "--disable isaacsim.robot.policy.examples",
    "--disable omni.kit.livestream.core",
    "--disable omni.kit.livestream.webrtc",
)

_RTX_CAPTURE_ARGS = (
    "/isaac-sim/apps/isaacsim.exp.base.kit",
    "--allow-root",
    "--no-window",
    "--/app/window/hideUi=true",
    "--/app/file/ignoreUnsavedOnExit=true",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
    "--/app/hydraEngine/waitIdle=true",
    "--/renderer/multiGpu/enabled=false",
    "--/rtx/multiGpu/enabled=false",
    "--/rtx/denoising/enabled=false",
    "--/rtx-transient/dlssg/enabled=false",
    "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false",
    "--/ngx/enabled=false",
    *_HEAVY_EXTENSION_DISABLE_ARGS,
)

_MINIMAL_CAPTURE_HEAD_ARGS = (
    "--allow-root",
    "--no-window",
    "--ext-folder /isaac-sim/kit/exts",
    "--ext-folder /isaac-sim/kit/extscore",
    "--ext-folder /isaac-sim/exts",
    "--ext-folder /isaac-sim/extscache",
    "--ext-folder /isaac-sim/apps",
    "--enable omni.usd",
    "--enable omni.kit.renderer.init",
    "--enable omni.kit.renderer.core",
    "--enable omni.kit.renderer.capture",
)

_MINIMAL_RTX_RENDERER_ARGS = (
    "--enable omni.hydra.rtx",
    "--/renderer/multiGpu/enabled=false",
    "--/rtx/multiGpu/enabled=false",
    "--/rtx/denoising/enabled=false",
    "--/rtx-transient/dlssg/enabled=false",
    "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false",
)

_MINIMAL_STORM_RENDERER_ARGS = (
    "--enable omni.kit.viewport.pxr",
    "--/renderer/enabled='pxr'",
    "--/renderer/active='pxr'",
)

_MINIMAL_CAPTURE_TAIL_ARGS = (
    "--enable omni.kit.viewport.window",
    "--enable omni.kit.viewport.utility",
    "--/app/window/hideUi=true",
    "--/app/file/ignoreUnsavedOnExit=true",
    "--/app/extensions/registryEnabled=true",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
    "--/app/hydraEngine/waitIdle=true",
    "--/ngx/enabled=false",
)


def _capture_python_script(
    *,
    label: str,
    output_path: str,
    ok_marker: str,
    warmup_steps: int,
    poll_steps: int,
    guarded_import: bool,
) -> str:
    import_block = _capture_import_block(guarded=guarded_import)
    return f"""\
from pathlib import Path
import time

import omni.kit.app
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

app = omni.kit.app.get_app()
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()

cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.CreateSizeAttr(1.0)
cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
light.CreateIntensityAttr(5000.0)
light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.AddTranslateOp().Set(Gf.Vec3d(0.0, -4.0, 2.0))
camera.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 0.0))

for idx in range({warmup_steps}):
    app.update()
    print("{label}_WARMUP", idx)

{import_block}

viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("no active viewport")

viewport.camera_path = "/World/Camera"
if hasattr(viewport, "resolution"):
    viewport.resolution = (320, 240)

for idx in range({warmup_steps}):
    app.update()
    print("{label}_VIEWPORT_UPDATE", idx)

out = Path("{output_path}")
result = capture_viewport_to_file(viewport, str(out))
print("{label}_RESULT", result)

for idx in range({poll_steps}):
    app.update()
    if out.exists() and out.stat().st_size > 0:
        print("{ok_marker}", out, out.stat().st_size)
        app.post_quit()
        raise SystemExit(0)
    time.sleep(0.05)

raise RuntimeError(f"capture file not produced: {{out}}")
"""


def _capture_import_block(*, guarded: bool) -> str:
    if not guarded:
        return "from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file"
    return """\
from omni.kit.viewport.utility import get_active_viewport

try:
    from omni.kit.viewport.utility import capture_viewport_to_file
except Exception as exc:
    raise RuntimeError(f"capture_viewport_to_file import failed: {exc!r}") from exc
"""


def _kit_command(script_path: str, script: str, args: tuple[str, ...], *, timeout_seconds: int = 240) -> str:
    separator = " " + "\\" + "\n  "
    kit_args = separator.join(args)
    return f"""\
set -euxo pipefail
{_vk_env_script()}
cat > {script_path} <<'PY'
{script}PY

timeout {timeout_seconds}s /isaac-sim/kit/kit \
  {kit_args} \
  --exec {script_path}
"""


def _rtx_capture_args() -> tuple[str, ...]:
    return _RTX_CAPTURE_ARGS


def _minimal_capture_args(*, storm: bool) -> tuple[str, ...]:
    renderer_args = _MINIMAL_STORM_RENDERER_ARGS if storm else _MINIMAL_RTX_RENDERER_ARGS
    return _MINIMAL_CAPTURE_HEAD_ARGS + renderer_args + _MINIMAL_CAPTURE_TAIL_ARGS


def _official_kit_render_capture_command() -> str:
    script_path = "/tmp/yubo_kit_render_capture.py"
    script = _capture_python_script(
        label="OFFICIAL_KIT_RENDER",
        output_path="/tmp/yubo_kit_capture.png",
        ok_marker="OFFICIAL_KIT_RENDER_CAPTURE_OK",
        warmup_steps=10,
        poll_steps=120,
        guarded_import=True,
    )
    return _kit_command(script_path, script, _rtx_capture_args())


def _official_kit_bare_render_capture_command() -> str:
    script_path = "/tmp/yubo_kit_bare_render_capture.py"
    script = _capture_python_script(
        label="OFFICIAL_KIT_BARE_RENDER",
        output_path="/tmp/yubo_kit_bare_capture.png",
        ok_marker="OFFICIAL_KIT_BARE_RENDER_CAPTURE_OK",
        warmup_steps=20,
        poll_steps=160,
        guarded_import=False,
    )
    return _kit_command(script_path, script, _minimal_capture_args(storm=False))


def _official_kit_storm_render_capture_command() -> str:
    script_path = "/tmp/yubo_kit_storm_render_capture.py"
    script = _capture_python_script(
        label="OFFICIAL_KIT_STORM_RENDER",
        output_path="/tmp/yubo_kit_storm_capture.png",
        ok_marker="OFFICIAL_KIT_STORM_RENDER_CAPTURE_OK",
        warmup_steps=20,
        poll_steps=160,
        guarded_import=False,
    )
    return _kit_command(script_path, script, _minimal_capture_args(storm=True))
