"""Shared Modal image helpers for the sibling ``../enn`` checkout."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import modal

# Keep upload/build context small: exclude caches, git, and rust/target (~GB).
ENN_MODAL_IGNORE: list[str] = [
    ".git",
    "_enn_data",
    "_kpop",
    "_malvin",
    "malvin_memories",
    ".malvin_memories",
    "target",
    "rust/target",
    "tests",
    "examples",
    "docs",
    "reports",
    "scripts",
    "admin",
    "**/debug",
    "**/release",
    "**/__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
    ".mypy_cache",
    ".venv",
    "**/*.whl",
]

ENN_MODAL_BUILD_COMMANDS: tuple[str, ...] = (
    ". $HOME/.cargo/env && "
    "export CARGO_BUILD_RUSTC_WRAPPER= && "
    "export RUSTFLAGS='-C link-arg=-Wl,--no-as-needed -C link-arg=-lopenblas' && "
    "cd /root/enn/rust/crates/enn-py && maturin build --release",
    "pip install $(find /root/enn/rust -path '*/wheels/*manylinux*.whl' | head -1) && pip install -e /root/enn",
)


def enn_project_root(project_root: Path) -> Path:
    return project_root.parent / "enn"


def add_enn_to_image(
    image: modal.Image,
    enn_root: Path,
    *,
    remote_path: str = "/root/enn",
) -> modal.Image:
    """Copy enn sources (excluding build artifacts) and build the PyO3 wheel in-image."""
    if not enn_root.is_dir():
        return image
    image = image.add_local_dir(
        str(enn_root),
        remote_path=remote_path,
        ignore=ENN_MODAL_IGNORE,
        copy=True,
    )
    return image.run_commands(*ENN_MODAL_BUILD_COMMANDS)
