import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HASH_EXCLUDE = frozenset({"problem_seed", "noise_seed_0", "num_reps"})
_DEFAULT_RESULTS = "results/uhd"
_APP_NAME = "yubo_uhd_batch"


def _config_hash(cfg: dict) -> str:
    d = {k: v for k, v in sorted(cfg.items()) if k not in _HASH_EXCLUDE}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _experiment_dir(results_dir: str, cfg: dict) -> Path:
    return Path(results_dir) / _config_hash(cfg)


def _trace_path(exp_dir: Path, i_rep: int) -> Path:
    return exp_dir / "traces" / f"{i_rep:05d}.jsonl"


def _gen_missing_reps(exp_dir: Path, num_reps: int, base_seed: int):
    for i in range(num_reps):
        tp = _trace_path(exp_dir, i)
        if tp.with_suffix(".done").exists():
            continue
        ps = base_seed + i
        yield i, ps, 10 * ps, tp


def _dict_to_toml(cfg: dict) -> str:
    lines = ["[uhd]"]
    for k, v in sorted(cfg.items()):
        if v is None:
            continue
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"{k} = {json.dumps(list(v))}")
    lines.append("")
    return "\n".join(lines)


def _parse_eval_lines(log_text: str) -> list[dict]:
    records = []
    for line in log_text.splitlines():
        s = line.strip()
        if not s.startswith("EVAL:"):
            continue
        parts = s[len("EVAL:") :].split()
        d: dict[str, str] = {}
        i = 0
        while i + 2 < len(parts):
            if parts[i + 1] == "=":
                d[parts[i]] = parts[i + 2]
                i += 3
            else:
                i += 1
        iter_key = "i_iter" if "i_iter" in d else "step" if "step" in d else None
        if iter_key is not None and "mu" in d:
            dt_prop = float(d["proposal_dt"]) if "proposal_dt" in d else 0.0
            dt_eval = float(d["eval_dt"]) if "eval_dt" in d else 0.0
            records.append(
                {
                    "i_iter": int(d[iter_key]),
                    "rreturn": float(d["mu"]),
                    "dt_prop": dt_prop,
                    "dt_eval": dt_eval,
                }
            )
    return records


def _run_subprocess(cfg: dict) -> tuple[str, int]:
    """Run exp_uhd.py local with a temp TOML. Returns (stdout, returncode)."""
    fd, tmp = tempfile.mkstemp(suffix=".toml")
    try:
        cfg = {k: v for k, v in cfg.items() if k != "num_reps"}
        with os.fdopen(fd, "w") as f:
            f.write(_dict_to_toml(cfg))
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "ops.exp_uhd", "local", tmp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            print(line, end="", flush=True)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        rc = proc.wait()
        if rc != 0:
            print(f"SUBPROCESS ERROR:\n{stderr}", file=sys.stderr)
        return "".join(stdout_lines), rc
    finally:
        os.unlink(tmp)


def _write_trace(tp: Path, records: list[dict]) -> None:
    tp.parent.mkdir(parents=True, exist_ok=True)
    with open(tp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tp.with_suffix(".done").touch()


def _write_config(exp_dir: Path, cfg: dict) -> None:
    config = {k: v for k, v in cfg.items() if k not in _HASH_EXCLUDE}
    config["opt_name"] = config.get("optimizer", "mezo")
    config["num_arms"] = 1
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _load_toml(path: str) -> dict:
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    section = data.get("uhd", data)
    return {k.replace("-", "_"): v for k, v in section.items()}


def _resolve_num_reps(cfg: dict, cli_num_reps: int | None) -> int:
    import click

    value = cfg.get("num_reps", 1) if cli_num_reps is None else cli_num_reps
    num_reps = int(value)
    if num_reps < 1:
        raise click.BadParameter("num_reps must be >= 1")
    return num_reps
