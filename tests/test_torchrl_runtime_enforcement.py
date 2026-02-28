from __future__ import annotations

import ast
from pathlib import Path

_TORCHRL_BACKEND_DIR = Path(__file__).resolve().parents[1] / "rl" / "torchrl"
_ALLOWED_SELECT_DEVICE_FILES = {
    "common.py",
    "runtime.py",
}


def _torchrl_algo_files() -> list[Path]:
    return sorted(path for path in _TORCHRL_BACKEND_DIR.rglob("*.py") if path.name != "__init__.py")


def test_no_direct_select_device_usage_outside_runtime_layer():
    violations: list[str] = []
    for path in _torchrl_algo_files():
        if path.name in _ALLOWED_SELECT_DEVICE_FILES:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rl.torchrl.common.common":
                if any(alias.name == "select_device" for alias in node.names):
                    violations.append(f"{path.relative_to(_TORCHRL_BACKEND_DIR)}: imports select_device from torchrl.common.common")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "select_device":
                    violations.append(f"{path.relative_to(_TORCHRL_BACKEND_DIR)}:{node.lineno}: direct select_device(...) call")
    assert not violations, "Direct select_device usage is forbidden outside runtime layer:\n" + "\n".join(violations)


def test_torchrl_trainers_use_shared_runtime_resolver():
    missing: list[str] = []
    for path in _torchrl_algo_files():
        if path.name in _ALLOWED_SELECT_DEVICE_FILES:
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        has_train_fn = any(isinstance(node, ast.FunctionDef) and node.name.startswith("train_") for node in tree.body)
        if not has_train_fn:
            continue
        uses_shared_resolver = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "resolve_torchrl_runtime":
                uses_shared_resolver = True
                break
            if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve_runtime":
                uses_shared_resolver = True
                break
        if not uses_shared_resolver:
            missing.append(str(path.relative_to(_TORCHRL_BACKEND_DIR)))
    assert not missing, "TorchRL trainer modules must use shared runtime resolution: " + ", ".join(missing)
