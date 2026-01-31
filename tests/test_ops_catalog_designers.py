import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_ops_catalog_designers_prints_sobol():
    script = _repo_root() / "ops" / "catalog.py"
    p = subprocess.run(
        [sys.executable, str(script), "designers"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "sobol" in p.stdout
