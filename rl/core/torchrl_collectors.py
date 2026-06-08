from __future__ import annotations


def collector_class(name: str):
    import torchrl.collectors as collectors

    cls = getattr(collectors, name, None)
    if cls is not None:
        return cls
    exported = __import__("torchrl.collectors.collectors", fromlist=[name])
    return getattr(exported, name)
