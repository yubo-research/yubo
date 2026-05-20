from __future__ import annotations

from ops.modal_nvidia_vulkan import nvidia_vulkan_icd_script


def _vk_env_script() -> str:
    return (
        nvidia_vulkan_icd_script()
        + """\
export __VK_LAYER_NV_optimus=NVIDIA_only
export __NV_PRIME_RENDER_OFFLOAD=1
"""
    )


def _probe_command(*, symlink_nvidia0: bool = False) -> str:
    symlink_script = ""
    if symlink_nvidia0:
        symlink_script = """\
if [ ! -e /dev/nvidia0 ]; then
  first_gpu="$(find /dev -maxdepth 1 -type c -name 'nvidia[0-9]*' | sort | head -1 || true)"
  if [ -n "${first_gpu}" ]; then ln -s "${first_gpu}" /dev/nvidia0; fi
fi
"""

    return f"""\
set -euxo pipefail
{_vk_env_script()}
{symlink_script}
nvidia-smi || true
printenv | sort | grep -E '^(NVIDIA|VK_|__NV|DISPLAY|XDG)' || true
ls -la /dev/nvidia* /dev/dri 2>/dev/null || true
find /usr/lib /usr/share /etc -iname '*nvidia*icd*' -o -iname '*vulkan*nvidia*' -o -iname 'libnvidia-vulkan*' 2>/dev/null | sort
cat /usr/share/vulkan/icd.d/nvidia_icd.json || true
ldconfig -p | grep -E 'nvidia|vulkan|EGL|GLX' || true
ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 || true
readelf -Ws /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep -E 'vk_icd|vkCreateInstance|vkGetInstanceProcAddr' || true
python - <<'PY'
import ctypes
for lib in ("libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libvulkan.so.1"):
    try:
        ctypes.CDLL(lib)
    except OSError as exc:
        print("CDLL_FAIL", lib, exc)
    else:
        print("CDLL_OK", lib)
PY
VK_LOADER_DEBUG=driver,error,warn vulkaninfo --summary || true
eglinfo -B || true
glxinfo -B || true
"""


def _isaacsim_smoke_command() -> str:
    return f"""\
set -euxo pipefail
{_vk_env_script()}
python -m pip install --disable-pip-version-check --extra-index-url https://pypi.nvidia.com 'isaacsim[all,extscache]==6.0.0.0'
python - <<'PY'
from isaacsim import SimulationApp

app = SimulationApp(
    {{
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": 640,
        "height": 480,
    }}
)
print("ISAACSIM_SIMULATION_APP_OK")
app.close()
PY
"""


def _official_probe_command() -> str:
    return f"""\
set -euxo pipefail
{_vk_env_script()}
nvidia-smi || true
printenv | sort | grep -E '^(CUDA|NVIDIA|VK_|__NV|DISPLAY|XDG|ISAAC)' || true
ls -la /dev/nvidia* /dev/dri 2>/dev/null || true
find /proc/driver/nvidia -maxdepth 5 -print 2>/dev/null || true
(find /usr/lib /usr/share /etc -iname '*nvidia*icd*' -o -iname '*vulkan*nvidia*' -o -iname 'libnvidia-vulkan*' 2>/dev/null | sort) || true
(find /usr/share/vulkan /etc/vulkan -maxdepth 4 -type f -print 2>/dev/null | sort) || true
cat /usr/share/vulkan/icd.d/nvidia_icd.json || true
ldconfig -p | grep -E 'nvidia|vulkan|EGL|GLX' || true
ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 || true
readelf -Ws /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 | grep -E 'vk_icd|vkCreateInstance|vkGetInstanceProcAddr' || true
VK_LOADER_DEBUG=driver,error,warn vulkaninfo --summary || true
eglinfo -B || true
glxinfo -B || true
"""


def _official_device_caps_command() -> str:
    return f"""\
set -euxo pipefail
{_vk_env_script()}
nvidia-smi -q | sed -n '1,220p' || true
printenv | sort | grep -E '^(CUDA|NVIDIA|VK_|__NV|DISPLAY|XDG)' || true
find /dev -maxdepth 3 \\( -name 'dri' -o -name 'nvidia*' \\) -print -exec ls -lad {{}} \\; 2>/dev/null | sort || true
find /dev/dri /dev/nvidia-caps -maxdepth 2 -print -exec ls -lad {{}} \\; 2>/dev/null | sort || true
find /proc/driver/nvidia -maxdepth 6 -print -exec sh -c 'for p do if [ -f "$p" ]; then echo "===== $p"; sed -n "1,120p" "$p" || true; fi; done' sh {{}} + 2>/dev/null || true
ls -l /proc/self/fd | sed -n '1,80p' || true
python - <<'PY'
from pathlib import Path

paths = [
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-modeset",
    "/dev/nvidia-caps",
    "/dev/nvidia-caps/nvidia-cap1",
    "/dev/nvidia-caps/nvidia-cap2",
    "/dev/dri",
    "/dev/dri/card0",
    "/dev/dri/renderD128",
]
for raw in paths:
    path = Path(raw)
    print("NODE", raw, "exists=", path.exists(), "is_char=", path.is_char_device(), "is_dir=", path.is_dir())
PY
vulkaninfo --summary || true
eglinfo -B || true
"""


def _official_isaacsim_smoke_command() -> str:
    return f"""\
set -euxo pipefail
{_vk_env_script()}
if [ -x /isaac-sim/python.sh ]; then
  isaac_python=/isaac-sim/python.sh
elif [ -x /isaac-sim/kit/python/bin/python3 ]; then
  isaac_python=/isaac-sim/kit/python/bin/python3
else
  isaac_python=python
fi
"${{isaac_python}}" - <<'PY'
from isaacsim import SimulationApp

app = SimulationApp(
    {{
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": 640,
        "height": 480,
    }}
)
print("OFFICIAL_ISAACSIM_SIMULATION_APP_OK")
app.close()
PY
"""


def _official_isaacsim_minimal_command(
    *,
    experience: str = "/isaac-sim/apps/isaacsim.exp.base.python.kit",
    disable_viewport_updates: bool = False,
) -> str:
    disable_viewport = "True" if disable_viewport_updates else "False"
    marker = "OFFICIAL_ISAACSIM_MINIMAL_NOVIEW_OK" if disable_viewport_updates else "OFFICIAL_ISAACSIM_MINIMAL_OK"
    return f"""\
set -euxo pipefail
{_vk_env_script()}
if [ -x /isaac-sim/python.sh ]; then
  isaac_python=/isaac-sim/python.sh
elif [ -x /isaac-sim/kit/python/bin/python3 ]; then
  isaac_python=/isaac-sim/kit/python/bin/python3
else
  isaac_python=python
fi
timeout 180s "${{isaac_python}}" - <<'PY'
from isaacsim import SimulationApp

config = {{
    "headless": True,
    "hide_ui": True,
    "active_gpu": 0,
    "physics_gpu": 0,
    "multi_gpu": False,
    "max_gpu_count": 1,
    "sync_loads": False,
    "width": 64,
    "height": 64,
    "window_width": 64,
    "window_height": 64,
    "renderer": "RaytracedLighting",
    "anti_aliasing": 0,
    "samples_per_pixel_per_frame": 1,
    "denoiser": False,
    "disable_viewport_updates": {disable_viewport},
    "enable_crashreporter": False,
    "fast_shutdown": True,
    "limit_cpu_threads": 17,
    "extra_args": [
        "--/renderer/multiGpu/enabled=false",
        "--/rtx/denoising/enabled=false",
        "--/rtx-transient/dlssg/enabled=false",
        "--/app/asyncRendering=false",
        "--/app/asyncRenderingLowLatency=false",
        "--/app/hydraEngine/waitIdle=true",
    ],
}}
app = SimulationApp(config, experience="{experience}")
print("OFFICIAL_ISAACSIM_MINIMAL_CONSTRUCTED")
for idx in range(3):
    app.update()
    print("OFFICIAL_ISAACSIM_MINIMAL_UPDATE", idx)
print("{marker}")
app.close()
PY
"""


def _official_kit_smoke_command(
    *,
    experience: str = "/isaac-sim/apps/isaacsim.exp.base.kit",
    disable_heavy_extensions: bool = False,
) -> str:
    disable_ext_args = ""
    marker = "OFFICIAL_KIT_SMOKE_OK"
    if disable_heavy_extensions:
        marker = "OFFICIAL_KIT_SMOKE_LIGHT_OK"
        disable_ext_args = """\
  --disable omni.replicator.core \
  --disable omni.replicator.nv \
  --disable omni.replicator.replicator_yaml \
  --disable isaacsim.sensors.rtx \
  --disable isaacsim.robot.policy.examples \
  --disable omni.kit.livestream.core \
  --disable omni.kit.livestream.webrtc \
"""

    return f"""\
set -euxo pipefail
{_vk_env_script()}
cat > /tmp/yubo_kit_smoke.py <<'PY'
import omni.kit.app

app = omni.kit.app.get_app()
print("OFFICIAL_KIT_SCRIPT_START")
for idx in range(5):
    app.update()
    print("OFFICIAL_KIT_SCRIPT_UPDATE", idx)
print("{marker}")
app.post_quit()
PY

timeout 180s /isaac-sim/kit/kit \
  {experience} \
  --allow-root \
  --no-window \
  --/app/window/hideUi=true \
  --/app/file/ignoreUnsavedOnExit=true \
  --/app/asyncRendering=false \
  --/app/asyncRenderingLowLatency=false \
  --/app/hydraEngine/waitIdle=true \
  --/renderer/multiGpu/enabled=false \
  --/rtx/multiGpu/enabled=false \
  --/rtx/denoising/enabled=false \
  --/rtx-transient/dlssg/enabled=false \
  --/rtx-transient/resourcemanager/enableTextureStreaming=false \
  --/rtx-transient/resourcemanager/enableGeometryStreaming=false \
  --/ngx/enabled=false \
  {disable_ext_args} \
  --exec /tmp/yubo_kit_smoke.py
"""


def _official_kit_help_command() -> str:
    return f"""\
set -euxo pipefail
{_vk_env_script()}
/isaac-sim/kit/kit --help | head -200
"""


def _official_isaaclab_rollout_video_command(
    *,
    env_tag: str = "isaaclab:Isaac-Velocity-Rough-Anymal-C-Direct-v0",
    video_prefix: str = "isaaclab-rollout",
    seed: int = 0,
) -> str:
    return f"""\
set -euxo pipefail
{_vk_env_script()}
python - <<'PY'
from pathlib import Path
import numpy as np

from common.video_rollout import rollout_episode
from problems.isaaclab_env_adapters import make_isaaclab_env

class ZeroPolicy:
    def __call__(self, obs):
        _ = obs
        return np.zeros((2,), dtype=np.float32)

class _Conf:
    env_name = "{env_tag}"
    max_steps = 32

    def make(self, render_mode=None):
        return make_isaaclab_env(
            self.env_name,
            headless=True,
            num_envs=1,
            render_mode=render_mode,
        )

video_dir = Path("/tmp/isaaclab-video")
video_dir.mkdir(parents=True, exist_ok=True)
ret = rollout_episode(
    _Conf(),
    ZeroPolicy(),
    seed={seed},
    render_video=True,
    video_dir=video_dir,
    video_prefix="{video_prefix}",
)
print("OFFICIAL_ISAACLAB_ROLLOUT_RETURN", ret)
print("OFFICIAL_ISAACLAB_VIDEO_DIR", video_dir)
for path in sorted(video_dir.glob("*")):
    print("OFFICIAL_ISAACLAB_VIDEO_FILE", path, path.stat().st_size)
PY
"""


def _official_inspect_command() -> str:
    return """\
set -euxo pipefail
find /isaac-sim/apps -maxdepth 1 -type f -name '*.kit' -print | sort
python - <<'PY'
from pathlib import Path

path = Path("/isaac-sim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py")
print(f"SIMULATION_APP_SOURCE={path}")
text = path.read_text()
for needle in (
    "DEFAULT_LAUNCHER_CONFIG",
    "def __init__",
    "extra_args",
    "experience",
    "renderer",
    "multi_gpu",
    "physics_gpu",
):
    print(f"\\n--- {needle} ---")
    idx = text.find(needle)
    if idx >= 0:
        start = max(0, idx - 1200)
        end = min(len(text), idx + 2400)
        print(text[start:end])
PY
"""
