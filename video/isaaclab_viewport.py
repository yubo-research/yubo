from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

_VIDEO_EYE = (0.0, -5.0, 1.35)
_VIDEO_TARGET = (0.0, 0.0, 0.85)


def _add_video_floor(stage: Any) -> None:
    from pxr import Gf, UsdGeom

    floor = UsdGeom.Cube.Define(stage, "/World/YuboVideoFloor")
    xform = UsdGeom.Xformable(floor)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.035))
    xform.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.02))
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.46, 0.48)])


def _set_camera_look_at(camera: Any) -> None:
    from pxr import Gf, UsdGeom

    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*_VIDEO_EYE),
        Gf.Vec3d(*_VIDEO_TARGET),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())


def prepare_isaaclab_video_view(app: Any, env: Any, *, width: int = 1280, height: int = 720) -> None:
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom, UsdLux
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    _add_video_floor(stage)

    camera_path = "/World/YuboVideoCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    _set_camera_look_at(camera)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    viewer = getattr(getattr(env, "cfg", None), "viewer", None)
    if viewer is not None:
        viewer.cam_prim_path = camera_path
        viewer.resolution = (int(width), int(height))
        viewer.eye = _VIDEO_EYE
        viewer.lookat = _VIDEO_TARGET

    light = UsdLux.DistantLight.Define(stage, "/World/YuboVideoLight")
    light_xform = UsdGeom.Xformable(light)
    light_xform.ClearXformOpOrder()
    light.CreateIntensityAttr(5000.0)
    light_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "set_camera_view"):
        sim.set_camera_view(eye=list(_VIDEO_EYE), target=list(_VIDEO_TARGET))

    viewport = get_active_viewport()
    if viewport is None:
        return
    viewport.camera_path = camera_path
    if hasattr(viewport, "resolution"):
        viewport.resolution = (int(width), int(height))

    for _ in range(4):
        app.update()


def capture_isaaclab_viewport_frame(app: Any, env: Any, output_path: Path, *, timeout_steps: int = 80) -> np.ndarray:
    from imageio.v2 import imread
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    prepare_isaaclab_video_view(app, env)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("IsaacLab viewport capture failed: no active viewport")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    capture_viewport_to_file(viewport, str(output_path))
    for _ in range(int(timeout_steps)):
        app.update()
        if output_path.exists() and output_path.stat().st_size > 0:
            frame = np.asarray(imread(output_path))
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            return frame
        time.sleep(0.02)
    raise RuntimeError(f"IsaacLab viewport capture failed: no frame written to {output_path}")
