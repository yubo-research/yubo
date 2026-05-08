from __future__ import annotations

from dataclasses import dataclass


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
