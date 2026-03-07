"""Types for designer specification and catalog."""

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True, slots=True)
class DesignerOptionSpec:
    name: str
    required: bool
    value_type: str
    description: str
    example: str
    allowed_values: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class DesignerCatalogEntry:
    base_name: str
    options: list[DesignerOptionSpec]
    dispatch: object


class DesignerSpec(NamedTuple):
    base: str
    general: dict
    specific: dict
