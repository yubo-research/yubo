from __future__ import annotations

from dataclasses import dataclass

from .designer_option_spec import DesignerOptionSpec


@dataclass(frozen=True, slots=True)
class DesignerCatalogEntry:
    base_name: str
    options: list[DesignerOptionSpec]
    dispatch: object
