from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

DEFAULT_VIDEO_GLOB = "runs/**/traces/videos/*.mp4"
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def logged_command(
    cmd: list[str],
    *,
    cwd: str = "/root",
    log_prefix: str,
    extra_env: dict[str, str] | None = None,
) -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[{log_prefix}] $ {printable}", flush=True)
    env = os.environ.copy()
    env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
    return_code = proc.wait()
    print(f"[{log_prefix}] exit={return_code} cmd={printable}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {printable}")
    return return_code


def collect_artifacts(patterns: list[str], *, root: str = "/root", log_prefix: str) -> list[tuple[str, bytes]]:
    root_path = Path(root)
    seen: set[Path] = set()
    artifacts: list[tuple[str, bytes]] = []
    total_bytes = 0

    for pattern in patterns:
        for path in sorted(root_path.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            data = path.read_bytes()
            total_bytes += len(data)
            if total_bytes > MAX_ARTIFACT_BYTES:
                raise RuntimeError(f"artifact export exceeds {MAX_ARTIFACT_BYTES} bytes; use a Modal Volume or narrower --export-glob")
            artifacts.append((_artifact_relpath(path, root=root_path), data))

    print(f"[{log_prefix}] collected_artifacts={len(artifacts)}", flush=True)
    for relpath, data in artifacts:
        print(f"[{log_prefix}] artifact {relpath} {len(data)} bytes", flush=True)
    return artifacts


def parse_export_globs(export_glob: str, *, export_videos: bool) -> list[str]:
    patterns = [part.strip() for part in export_glob.split(",") if part.strip()]
    if export_videos and DEFAULT_VIDEO_GLOB not in patterns:
        patterns.append(DEFAULT_VIDEO_GLOB)
    return patterns


def write_artifacts(artifacts: list[tuple[str, bytes]], *, export_dir: str, log_prefix: str) -> None:
    out_root = Path(export_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for relpath, data in artifacts:
        out_path = out_root / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"[{log_prefix}] wrote {out_path} {len(data)} bytes", flush=True)


def _artifact_relpath(path: Path, *, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return path.name
