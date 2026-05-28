"""Small helpers shared by Modal batch implementations."""

from __future__ import annotations


def delete_keys_from_dicts(keys, *dicts) -> None:
    for key in keys:
        for dct in dicts:
            try:
                del dct[key]
            except KeyError:
                pass
