from __future__ import annotations

try:
    from jaxtyping import Array, Float, PRNGKeyArray
except ImportError:  # jaxtyping < 0.3
    from jaxtyping import Float, Shaped

    Array = Shaped
    PRNGKeyArray = Shaped

__all__ = ["Array", "Float", "PRNGKeyArray"]
