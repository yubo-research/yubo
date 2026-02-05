#!/usr/bin/env python3
import ast
from dataclasses import dataclass
from pathlib import Path

EXCLUDE_TOP_DIRS = {
    ".git",
    "__pycache__",
    "admin",
    "analysis",
    "data",
    "experiments",
    "figures",
    "notes",
    "ops",
    "tests",
}

MAX_FAN_IN = 15
MAX_FAN_OUT = 8


@dataclass(frozen=True, slots=True)
class FanStats:
    fan_in: int
    fan_out: int


def _iter_python_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in repo_root.rglob("*.py"):
        rel = p.relative_to(repo_root)
        if not rel.parts:
            continue
        if rel.parts[0] in EXCLUDE_TOP_DIRS:
            continue
        if "__pycache__" in rel.parts:
            continue
        out.append(p)
    return out


def _module_name(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root)
    return ".".join(rel.with_suffix("").parts)


def _resolve_relative(from_mod: str, level: int, module: str | None) -> str | None:
    parts = from_mod.split(".")
    if level > len(parts):
        return None
    base = parts[:-level]
    if module:
        base += module.split(".")
    return ".".join(base)


def _shrink_to_internal(target: str, internal: set[str]) -> str | None:
    t = target
    while t and t not in internal:
        if "." in t:
            t = t.rsplit(".", 1)[0]
        else:
            t = ""
    return t or None


def compute_fan_stats(repo_root: Path) -> dict[str, FanStats]:
    files = _iter_python_files(repo_root)
    mod_by_file = {p: _module_name(repo_root, p) for p in files}
    internal = set(mod_by_file.values())

    edges: set[tuple[str, str]] = set()
    for p, from_mod in mod_by_file.items():
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    t = _shrink_to_internal(alias.name, internal)
                    if t:
                        edges.add((from_mod, t))
            elif isinstance(node, ast.ImportFrom):
                if node.module is None and node.level == 0:
                    continue
                base = _resolve_relative(from_mod, node.level, node.module) if node.level else node.module
                if not base:
                    continue
                t = _shrink_to_internal(base, internal)
                if t:
                    edges.add((from_mod, t))

    fan_out = {m: 0 for m in internal}
    fan_in = {m: 0 for m in internal}
    for a, b in edges:
        if a == b:
            continue
        fan_out[a] += 1
        fan_in[b] += 1

    return {m: FanStats(fan_in=fan_in[m], fan_out=fan_out[m]) for m in internal}


def check_fan_limits(repo_root: Path) -> list[str]:
    stats = compute_fan_stats(repo_root)
    violations: list[str] = []
    for mod, st in stats.items():
        if st.fan_in > MAX_FAN_IN:
            violations.append(f"fan_in {st.fan_in} > {MAX_FAN_IN}: {mod}")
        if st.fan_out > MAX_FAN_OUT:
            violations.append(f"fan_out {st.fan_out} > {MAX_FAN_OUT}: {mod}")
    return sorted(violations)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations = check_fan_limits(repo_root)
    if not violations:
        return 0
    print("Fan-in/fan-out constraints violated:")
    for v in violations:
        print(f"  - {v}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
