"""Types for designer specification and catalog."""

from typing import NamedTuple

from .designer_catalog_entry import DesignerCatalogEntry
from .designer_def import DesignerDef
from .designer_option_spec import DesignerOptionSpec


class DesignerSpec(NamedTuple):
    base: str
    general: dict
    specific: dict


__all__ = [
    "DesignerCatalogEntry",
    "DesignerDef",
    "DesignerOptionSpec",
    "DesignerSpec",
]
