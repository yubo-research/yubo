from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_local_smoke(tmp_path: Path):
    # Use a lightweight gym env (no MNIST download).
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        """
[uhd]
env_tag = "pend"
num_rounds = 1
""".lstrip()
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(repo_root / "ops" / "exp_uhd.py"),
            "local",
            str(cfg),
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "EVAL: i_iter = 0" in proc.stdout
