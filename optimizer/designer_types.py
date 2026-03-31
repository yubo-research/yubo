"""Types for designer specification and catalog."""

from dataclasses import dataclass
from typing import Callable, NamedTuple


@dataclass(frozen=True, slots=True)
class DesignerOptionSpec:
    name: str
    required: bool
    value_type: str
    description: str
    example_suffix: str
    allowed_values: tuple[str, ...] | None = None

    def example(self, designer_name: str) -> str:
        return f"{designer_name}/{self.example_suffix}"


@dataclass(frozen=True, slots=True)
class DesignerCatalogEntry:
    base_name: str
    options: list["DesignerOptionSpec"]
    dispatch: object


class DesignerSpec(NamedTuple):
    base: str
    general: dict
    specific: dict


@dataclass(frozen=True, slots=True)
class DesignerDef:
    name: str
    builder: Callable
    option_specs: tuple["DesignerOptionSpec", ...] = ()

    def has_options(self) -> bool:
        return len(self.option_specs) > 0
