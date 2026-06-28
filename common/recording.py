from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.mapping_keys import coerce_mapping_keys

RECORDING_SECTION = "recording"
RECORDING_KEYS = frozenset({"enabled", "episodes", "keep", "select", "seed", "prefix"})


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False
    episodes: int = 8
    keep: int = 3
    select: str = "best"
    seed: int | None = None
    prefix: str = "bo"

    @classmethod
    def from_raw(cls, raw: Any, *, source: str) -> "RecordingConfig":
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"{source} must be a table.")
        cfg = coerce_mapping_keys(
            raw,
            source=source,
            valid_keys=RECORDING_KEYS,
            not_mapping_msg=f"{source} must be a table.",
        )
        episodes = int(cfg.get("episodes", 8))
        keep = int(cfg.get("keep", 3))
        if episodes < 1:
            raise ValueError(f"{source}.episodes must be >= 1 (got: {episodes})")
        if keep < 0:
            raise ValueError(f"{source}.keep must be >= 0 (got: {keep})")
        select = str(cfg.get("select", "best")).strip().lower()
        if select not in {"best", "first", "random"}:
            raise ValueError(f"{source}.select must be one of: best, first, random (got: {select!r})")
        seed_raw = cfg.get("seed")
        seed = None if seed_raw in (None, "None", "") else int(seed_raw)
        return cls(
            enabled=_true_false(cfg.get("enabled", False)),
            episodes=episodes,
            keep=keep,
            select=select,
            seed=seed,
            prefix=str(cfg.get("prefix", "bo")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _true_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    string_bool = str(value).strip().lower()
    if string_bool in {"false", "f"}:
        return False
    if string_bool in {"true", "t"}:
        return True
    raise ValueError(f"expected boolean value, got {value!r}")
