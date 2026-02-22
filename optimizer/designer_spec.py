from dataclasses import dataclass
from typing import NamedTuple

from .designer_errors import NoSuchDesignerError

_GENERAL_OPT_KEYS = {"num_keep", "keep_style", "model_spec", "sample_around_best"}


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


def _parse_opt_value(raw: str):
    s = raw.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_slash_opts(name: str) -> tuple[str, dict]:
    parts = [p for p in name.split("/") if p != ""]
    if not parts:
        raise NoSuchDesignerError("Empty designer name")
    base = parts[0]
    opts: dict[str, object] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise NoSuchDesignerError(f"Invalid designer option '{part}'. Expected 'key=value' in '{name}'.")
        k, v = part.split("=", 1)
        k = k.strip()
        if not k:
            raise NoSuchDesignerError(f"Invalid designer option '{part}'. Empty key in '{name}'.")
        if k in opts:
            raise NoSuchDesignerError(f"Duplicate option '{k}' in '{name}'.")
        opts[k] = _parse_opt_value(v)
    return base, opts


def parse_designer_spec(designer_name: str) -> DesignerSpec:
    parsed = _parse_options(designer_name)
    base_with_slash = parsed.designer_name
    general = {
        "num_keep": parsed.num_keep,
        "keep_style": parsed.keep_style,
        "model_spec": parsed.model_spec,
        "sample_around_best": parsed.sample_around_best,
    }

    base, slash_opts = _parse_slash_opts(base_with_slash)
    all_opts = dict(slash_opts)

    for k in _GENERAL_OPT_KEYS:
        if k in all_opts:
            general[k] = all_opts.pop(k)

    return DesignerSpec(base=base, general=general, specific=all_opts)


def _parse_options(designer_name):
    class _ParsedOptions(NamedTuple):
        designer_name: str
        num_keep: int | None
        keep_style: str | None
        model_spec: str | None
        sample_around_best: bool

    if ":" in designer_name:
        designer_name, options_str = designer_name.split(":")
        options = options_str.split("-")
    else:
        options = []

    num_keep = None
    keep_style = None
    model_spec = None
    sample_around_best = False

    keep_style_map = {
        "s": "some",
        "b": "best",
        "r": "random",
        "t": "trailing",
        "p": "lap",
    }

    for option in options:
        if option[0] == "K":
            keep_style = keep_style_map.get(option[1])
            assert keep_style is not None, option
            num_keep = int(option[2:])
            print(f"OPTION: num_keep = {num_keep} keep_style = {keep_style}")
        elif option[0] == "M":
            model_spec = option[1:]
            print(f"OPTION model_spec = {option}")
        elif option[0] == "O":
            if option[1:] == "sab":
                sample_around_best = True
        else:
            assert False, ("Unknown option", option)

    return _ParsedOptions(
        designer_name=designer_name,
        num_keep=num_keep,
        keep_style=keep_style,
        model_spec=model_spec,
        sample_around_best=sample_around_best,
    )
