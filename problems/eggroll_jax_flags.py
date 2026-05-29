from __future__ import annotations


def eggroll_jax_sim_enabled(opt_name: str | None) -> bool:
    """True when EggRoll should use the JAX path for Isaac Lab env tags."""
    text = str(opt_name or "").strip()
    if not text:
        return False
    from optimizer.designer_spec import parse_designer_spec

    try:
        spec = parse_designer_spec(text)
    except Exception:
        return False
    if str(spec.base) != "eggroll":
        return False
    return bool(spec.specific.get("jax_sim", False))
