"""Coverage for ennbo source patch used by admin/pixi_extras_mac.sh."""

from __future__ import annotations

from admin.patch_enn_failure_tolerance_dim import _MARKER, patch_enn_repo


def test_patch_enn_failure_tolerance_dim_exports() -> None:
    assert "failure_tolerance_dim" in _MARKER
    assert callable(patch_enn_repo)
