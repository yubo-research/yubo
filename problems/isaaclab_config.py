from __future__ import annotations

from typing import Any

ISAACLAB_DEFAULT_KIT_ARGS = (
    "--no-window "
    "--/app/window/hideUi=true "
    "--/app/file/ignoreUnsavedOnExit=true "
    "--/app/asyncRendering=false "
    "--/app/asyncRenderingLowLatency=false "
    "--/app/hydraEngine/waitIdle=true "
    "--/renderer/multiGpu/enabled=false "
    "--/rtx/multiGpu/enabled=false "
    "--/rtx/denoising/enabled=false "
    "--/rtx-transient/dlssg/enabled=false "
    "--/rtx-transient/resourcemanager/enableTextureStreaming=false "
    "--/rtx-transient/resourcemanager/enableGeometryStreaming=false"
)


def disable_command_debug_visualizers(cfg: Any) -> None:
    commands = getattr(cfg, "commands", None)
    if commands is None:
        return
    names = commands if isinstance(commands, dict) else dir(commands)
    for name in names:
        if str(name).startswith("_"):
            continue
        term = commands[name] if isinstance(commands, dict) else getattr(commands, name, None)
        if hasattr(term, "debug_vis"):
            setattr(term, "debug_vis", False)
