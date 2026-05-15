"""Compose designer dispatch tables from split registry modules."""

from functools import partial

from .designer_errors import NoSuchDesignerError
from .designer_registry_builders import _build_turbo_enn, _build_turbo_enn_py, _no_opts
from .designer_registry_context import _SimpleContext
from .designer_registry_defs import _DESIGNER_DEFS, _DESIGNER_OPTION_SPECS
from .designer_registry_option_handlers import _build_turbo_enn_f, _d_turbo_enn_fit_ucb
from .designer_registry_simple_table import _SIMPLE_BUILDERS, _SIMPLE_DISPATCH
from .designer_types import DesignerDef


def _wrap_designer_def(d: DesignerDef):
    """Wrap builder with _no_opts if designer has no option specs."""
    if d.option_specs:
        return d.builder
    return partial(_no_opts, d.name, d.builder)


def _d_turbo_enn_simple(ctx, opts: dict, *, kind: str):
    if opts:
        raise NoSuchDesignerError(f"Designer '{kind}' does not support option(s): {', '.join(sorted(opts))}.")
    return _build_turbo_enn(ctx, "turbo-enn-p" if kind == "turbo-enn" else kind)


def _d_turbo_enn_py_simple(ctx, opts: dict, *, kind: str):
    if opts:
        raise NoSuchDesignerError(f"Designer '{kind}' does not support option(s): {', '.join(sorted(opts))}.")
    return _build_turbo_enn_py(ctx, kind)


_TURBO_OPTION_DISPATCH = {
    "turbo-enn": partial(_d_turbo_enn_simple, kind="turbo-enn"),
    "turbo-enn-p": partial(_d_turbo_enn_simple, kind="turbo-enn-p"),
    "turbo_py-enn-p": partial(_d_turbo_enn_py_simple, kind="turbo_py-enn-p"),
    "turbo_py-enn-fit-ucb": partial(_d_turbo_enn_py_simple, kind="turbo_py-enn-fit-ucb"),
    "turbo-zero": partial(_d_turbo_enn_simple, kind="turbo-zero"),
    "turbo-one": partial(_d_turbo_enn_simple, kind="turbo-one"),
    "turbo-one-nds": partial(_d_turbo_enn_simple, kind="turbo-one-nds"),
    "turbo-one-ucb": partial(_d_turbo_enn_simple, kind="turbo-one-ucb"),
    "lhd_only": partial(_d_turbo_enn_simple, kind="lhd_only"),
    "morbo-zero": partial(_d_turbo_enn_simple, kind="morbo-zero"),
    "morbo-one": partial(_d_turbo_enn_simple, kind="morbo-one"),
    "morbo-enn": partial(_d_turbo_enn_simple, kind="morbo-enn"),
}


_DESIGNER_DISPATCH = (
    {name: partial(_no_opts, name, builder) for name, builder in _SIMPLE_BUILDERS.items()}
    | {d.name: _wrap_designer_def(d) for d in _DESIGNER_DEFS}
    | _TURBO_OPTION_DISPATCH
)

__all__ = [
    "_DESIGNER_DISPATCH",
    "_DESIGNER_OPTION_SPECS",
    "_SIMPLE_DISPATCH",
    "_SimpleContext",
    "_build_turbo_enn",
    "_build_turbo_enn_f",
    "_d_turbo_enn_fit_ucb",
]
