from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .designer_option_spec import DesignerOptionSpec


@dataclass(frozen=True, slots=True)
class DesignerDef:
    name: str
    builder: Callable
    option_specs: tuple[DesignerOptionSpec, ...] = ()

    def has_options(self) -> bool:
        return len(self.option_specs) > 0
