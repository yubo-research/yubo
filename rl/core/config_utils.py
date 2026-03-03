from __future__ import annotations

import dataclasses
from typing import Any, ClassVar


def dataclass_config_from_dict(
    config_cls,
    raw: dict[str, Any],
    *,
    tuple_int_keys: tuple[str, ...] = (),
    int_keys: tuple[str, ...] = (),
):
    data = {key: value for key, value in raw.items() if key in {field.name for field in dataclasses.fields(config_cls)}}
    for key in tuple_int_keys:
        if key in data and data[key] is not None:
            data[key] = tuple((int(x) for x in data[key]))
    for key in int_keys:
        if key in data and data[key] is not None:
            data[key] = int(data[key])
    return config_cls(**data)


class DataclassFromDictMixin:
    _tuple_int_keys: ClassVar[tuple[str, ...]] = ()
    _int_keys: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]):
        return dataclass_config_from_dict(
            cls,
            raw,
            tuple_int_keys=tuple(cls._tuple_int_keys),
            int_keys=tuple(cls._int_keys),
        )
