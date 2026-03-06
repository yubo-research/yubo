"""Types for designer parsing."""

from typing import NamedTuple


class ParsedOptions(NamedTuple):
    designer_name: str
    num_keep: int | None
    keep_style: str | None
    model_spec: str | None
    sample_around_best: bool
