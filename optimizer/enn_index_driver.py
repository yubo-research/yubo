from __future__ import annotations

from enn.turbo.config.enn_index_driver import ENNIndexDriver


def parse_enn_index_driver(name: str) -> ENNIndexDriver:
    value = str(name).lower()
    if value == "hnsw":
        return ENNIndexDriver.HNSW
    if value == "hnsw_disk":
        return getattr(ENNIndexDriver, "HNSW_DISK", ENNIndexDriver.HNSW)
    return ENNIndexDriver.FLAT


__all__ = ["parse_enn_index_driver"]
