"""Backward-compatible re-export; prefer :class:`UHDDriver`."""

from optimizer.uhd_driver import UHDDriver

UHDLoop = UHDDriver

__all__ = ["UHDLoop", "UHDDriver"]
