"""Compose designer dispatch tables from split registry modules."""

from functools import partial

from .designer_registry_builders import _build_turbo_enn, _no_opts
from .designer_registry_context import _SimpleContext
from .designer_registry_defs import _DESIGNER_DEFS, _DESIGNER_OPTION_SPECS
from .designer_registry_option_handlers import _build_turbo_enn_f, _d_turbo_enn_fit_ucb, _d_turbo_enn_p
from .designer_registry_simple_table import _SIMPLE_BUILDERS, _SIMPLE_DISPATCH
from .designer_types import DesignerDef


def _wrap_designer_def(d: DesignerDef):
    """Wrap builder with _no_opts if designer has no option specs."""
    if d.option_specs:
        return d.builder
    return partial(_no_opts, d.name, d.builder)


_DESIGNER_DISPATCH = {name: partial(_no_opts, name, builder) for name, builder in _SIMPLE_BUILDERS.items()} | {
    d.name: _wrap_designer_def(d) for d in _DESIGNER_DEFS
}

__all__ = [
    "_DESIGNER_DISPATCH",
    "_DESIGNER_OPTION_SPECS",
    "_SIMPLE_DISPATCH",
    "_SimpleContext",
    "_build_turbo_enn",
    "_build_turbo_enn_f",
    "_d_turbo_enn_fit_ucb",
    "_d_turbo_enn_p",
]
