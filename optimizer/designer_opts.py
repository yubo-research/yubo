from __future__ import annotations

from .designer_errors import NoSuchDesignerError


class OptParser:
    @staticmethod
    def require_int(opts: dict, key: str, *, example: str) -> int:
        if key not in opts:
            raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
        v = opts[key]
        if not isinstance(v, int):
            raise NoSuchDesignerError(f"Designer option '{key}' must be an int.")
        return v

    @staticmethod
    def require_str_in(opts: dict, key: str, allowed: set[str], *, example: str) -> str:
        if key not in opts:
            raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
        v = opts[key]
        if not isinstance(v, str):
            raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
        if v not in allowed:
            raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
        return v

    @staticmethod
    def optional_int(opts: dict, key: str) -> int | None:
        if key not in opts:
            return None
        v = opts[key]
        if not isinstance(v, int):
            raise NoSuchDesignerError(f"Designer option '{key}' must be an int.")
        return v

    @staticmethod
    def optional_float(opts: dict, key: str) -> float | None:
        if key not in opts:
            return None
        v = opts[key]
        if not isinstance(v, (int, float)):
            raise NoSuchDesignerError(f"Designer option '{key}' must be a float.")
        return float(v)

    @staticmethod
    def optional_str_in(opts: dict, key: str, allowed: set[str]) -> str | None:
        if key not in opts:
            return None
        v = opts[key]
        if not isinstance(v, str):
            raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
        if v not in allowed:
            raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
        return v
