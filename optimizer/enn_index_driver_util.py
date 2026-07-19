"""Shared ENN index-driver string → enum resolution for designers."""


def resolve_enn_index_driver(index_driver: str | None):
    """Map optional designer string to ``ENNIndexDriver`` enum."""
    ENNIndexDriver = __import__("enn.turbo.config.enn_index_driver", fromlist=["ENNIndexDriver"]).ENNIndexDriver
    if index_driver is None:
        return ENNIndexDriver.FLAT
    v = index_driver.lower()
    if v in ("flat", "exact"):
        return ENNIndexDriver.FLAT
    if v in ("bpann_disk", "bpanndisk"):
        return ENNIndexDriver.BPANN_DISK
    raise ValueError(f"Invalid index_driver: {index_driver}")
