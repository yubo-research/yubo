#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
import subprocess
import sys

import modal

from ops.modal_hyperscalees_image import RUNTIME_GEMMA4_MTP, mk_image, selected_modal_runtime

app = modal.App(name="yubo-hyperscalees")
_RUNTIME = selected_modal_runtime(sys.argv)
image = mk_image(modal, runtime=_RUNTIME)

_TIMEOUT_SECONDS = 24 * 60 * 60


def _logged_command(cmd: list[str], *, cwd: str = "/root") -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[modal-hyperscalees] $ {printable}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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
    print(f"[modal-hyperscalees] exit={return_code} cmd={printable}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {printable}")
    return return_code


@app.function(image=image, gpu="L4", timeout=_TIMEOUT_SECONDS)
def run_hyperscalees_command(command: str) -> str:
    _logged_command(["bash", "-lc", command])
    return "ok"


def _preflight_command() -> str:
    if _RUNTIME == RUNTIME_GEMMA4_MTP:
        return "set -euxo pipefail; nvidia-smi; python -c " + shlex.quote(
            "import sys, torch, vllm; "
            "print(sys.version); "
            "print('torch', torch.__version__, torch.version.cuda); "
            "print('vllm', getattr(vllm, '__version__', 'unknown'))"
        )
    return "set -euxo pipefail; nvidia-smi; micromamba env list; micromamba run -n yubo-hyperscalees python -c " + shlex.quote(
        "import hyperscalees, sys, torch; "
        "print(sys.version); "
        "print('torch', torch.__version__, torch.version.cuda); "
        "print('hyperscalees', hyperscalees.__name__)"
    )


@app.local_entrypoint()
def main(command: str = "preflight", runtime: str = _RUNTIME) -> None:
    if runtime != _RUNTIME:
        raise ValueError(f"Modal runtime is selected while loading the app. Use --runtime {_RUNTIME!r} for this invocation.")
    if command == "preflight":
        command = _preflight_command()
    print(f"[modal-hyperscalees] runtime={_RUNTIME} command={command!r}", flush=True)
    run_hyperscalees_command.remote(command)
