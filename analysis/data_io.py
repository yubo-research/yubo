import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TraceRecord:
    i_iter: int
    dt_prop: float
    dt_eval: float
    rreturn: float
    env_name: Optional[str] = None
    opt_name: Optional[str] = None


@contextmanager
def data_writer(out_fn):
    if out_fn != "-":
        f = open(out_fn, "w")
    else:
        f = sys.stdout
    try:
        yield f
    finally:
        if f != sys.stdout:
            f.close()


def data_is_done(fn, quiet=True):
    done_marker = Path(fn).with_suffix(".done")
    if done_marker.exists():
        return True

    if not os.path.exists(fn):
        return False
    with open(fn, "rb") as f:
        try:
            f.seek(-5, 2)
        except OSError:
            return False
        x = f.read(5)

        if x != b"DONE\n":
            return False
        f.seek(0, 0)
        if not quiet:
            for line in f:
                print(line.strip().decode())
        return True


def mark_done(fn):
    done_marker = Path(fn).with_suffix(".done")
    done_marker.touch()


def write_config(out_dir: str, config: dict):
    config_path = Path(out_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def read_config(out_dir: str) -> dict:
    config_path = Path(out_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def write_trace_jsonl(trace_fn: str, records: list[TraceRecord]):
    Path(trace_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(trace_fn, "w") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")
    mark_done(trace_fn)


def read_trace_jsonl(trace_fn: str) -> list[TraceRecord]:
    records = []
    with open(trace_fn) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                records.append(TraceRecord(**d))
    return records


def write_log_jsonl(log_fn: str, lines: list[str]):
    Path(log_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(log_fn, "w") as f:
        for line in lines:
            f.write(json.dumps({"log": line}) + "\n")
