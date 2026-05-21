from __future__ import annotations

from importlib import import_module


def patch_replicator_seed_without_graph() -> None:
    try:
        rep = import_module("omni.replicator.core")
    except Exception:
        return

    set_global_seed = getattr(rep, "set_global_seed", None)
    if not callable(set_global_seed) or getattr(set_global_seed, "_yubo_isaaclab_seed_patch", False):
        return

    def safe_set_global_seed(seed):
        try:
            return set_global_seed(seed)
        except ValueError as exc:
            if str(exc) == "Unable to retrieve replicator graph":
                return None
            raise

    safe_set_global_seed._yubo_isaaclab_seed_patch = True
    rep.set_global_seed = safe_set_global_seed
