from __future__ import annotations

from typing import Any

from .designer_registry_builders import _load_symbol
from .designer_registry_context import _SimpleContext


def _build_eggroll(ctx: _SimpleContext, opts: dict[str, Any]):
    EggRollDesigner = _load_symbol("optimizer.eggroll_designer", "EggRollDesigner")
    return EggRollDesigner(ctx.policy, ctx.env_conf, **dict(opts))
