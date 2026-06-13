"""Shared MeZO two-phase helpers for torch and numpy optimizers."""


def skip_mezo_negative_pair(obj) -> None:
    if obj._positive_phase:
        raise RuntimeError("skip_negative is only valid after the positive phase")
    obj._positive_phase = True
    obj._seed += 1
