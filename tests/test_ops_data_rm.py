import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_ops_data_rm_results_dir_and_hash_deletes(tmp_path: Path):
    results_dir = tmp_path / "results"
    exp_hash = "edec8e1b"
    exp_dir = results_dir / exp_hash
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text('{"opt_name": "x", "env_tag": "y"}\n')
    (exp_dir / "traces").mkdir(parents=True)
    (exp_dir / "traces" / "00000.jsonl").write_text("{}\n")
    (exp_dir / "traces" / "00000.done").write_text("")

    script = _repo_root() / "ops" / "data.py"
    p = subprocess.run(
        [sys.executable, str(script), "rm", str(results_dir), exp_hash, "-f"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert not exp_dir.exists()


def test_ops_data_rm_multiple_hashes_deletes_all(tmp_path: Path):
    results_dir = tmp_path / "results"
    exp_hashes = ("a1b2c3d4", "edec8e1b")
    for h in exp_hashes:
        exp_dir = results_dir / h
        exp_dir.mkdir(parents=True)
        (exp_dir / "config.json").write_text('{"opt_name": "x", "env_tag": "y"}\n')
        (exp_dir / "traces").mkdir(parents=True)
        (exp_dir / "traces" / "00000.jsonl").write_text("{}\n")
        (exp_dir / "traces" / "00000.done").write_text("")

    script = _repo_root() / "ops" / "data.py"
    p = subprocess.run(
        [sys.executable, str(script), "rm", str(results_dir), *exp_hashes, "-f"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    for h in exp_hashes:
        assert not (results_dir / h).exists()


def test_ops_data_rm_rejects_path_traversal(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sentinel = results_dir / "sentinel.txt"
    sentinel.write_text("keep")

    script = _repo_root() / "ops" / "data.py"
    p = subprocess.run(
        [sys.executable, str(script), "rm", str(results_dir), "..", "-f"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode != 0
    assert sentinel.exists()


def test_ops_data_rm_bad_hash_aborts_all(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    good = results_dir / "edec8e1b"
    good.mkdir(parents=True)
    (good / "config.json").write_text('{"opt_name": "x", "env_tag": "y"}\n')

    script = _repo_root() / "ops" / "data.py"
    p = subprocess.run(
        [sys.executable, str(script), "rm", str(results_dir), "edec8e1b", "..", "-f"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode != 0
    assert good.exists()
