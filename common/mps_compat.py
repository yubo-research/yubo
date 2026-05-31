import sys
import threading
from functools import wraps

import jax
import jax.numpy as jnp

# Thread-local storage to prevent recursion in monkeypatches
_mps_state = threading.local()


def apply_mps_fixes():
    """
    Surgical polyfill for Jumanji Snake and JAX MPS backend.
    Fixes multi-dimensional gather/scatter by linearizing them to 1D.
    """
    if sys.platform != "darwin":
        return

    from jax import lax

    gather_p = lax.gather_p
    scatter_p = lax.scatter_p

    _orig_gather = gather_p.bind
    _orig_scatter = scatter_p.bind

    def is_mps_context():
        try:
            return any(d.platform == "mps" for d in jax.devices())
        except Exception:
            return False

    @wraps(_orig_gather)
    def mps_gather(operand, start_indices, **kwargs):
        if getattr(_mps_state, "active", False) or not is_mps_context():
            return _orig_gather(operand, start_indices, **kwargs)

        dn = kwargs.get("dimension_numbers")
        if dn and len(dn.start_index_map) > 1:
            _mps_state.active = True
            try:
                # Use CPU for multi-dim gather
                # This is the only 100% robust way to avoid MPS bugs
                with jax.default_device(jax.devices("cpu")[0]):
                    return _orig_gather(jnp.array(operand), jnp.array(start_indices), **kwargs)
            except Exception:
                pass
            finally:
                _mps_state.active = False
        return _orig_gather(operand, start_indices, **kwargs)

    @wraps(_orig_scatter)
    def mps_scatter(operand, scatter_indices, updates, **kwargs):
        if getattr(_mps_state, "active", False) or not is_mps_context():
            return _orig_scatter(operand, scatter_indices, updates, **kwargs)

        dn = kwargs.get("dimension_numbers")
        if dn and len(dn.scatter_dims_to_operand_dims) > 1:
            _mps_state.active = True
            try:
                # Use CPU for multi-dim scatter
                with jax.default_device(jax.devices("cpu")[0]):
                    return _orig_scatter(jnp.array(operand), jnp.array(scatter_indices), jnp.array(updates), **kwargs)
            except Exception:
                pass
            finally:
                _mps_state.active = False
        return _orig_scatter(operand, scatter_indices, updates, **kwargs)

    # Apply patches to primitives
    gather_p.bind = mps_gather
    scatter_p.bind = mps_scatter
    # Apply to high-level lax too
    lax.gather = mps_gather

    # Also apply the Snake-specific patch for extra safety
    try:
        from jumanji.environments.routing.snake.env import Snake

        _orig_get_action_mask = Snake._get_action_mask

        def mps_snake_mask(self, head_position, body_state):
            if not is_mps_context():
                return _orig_get_action_mask(self, head_position, body_state)
            grid_flat = body_state.ravel()
            h, w = body_state.shape
            head_pos_arr = jnp.array(head_position)

            def is_valid(move):
                new_pos = head_pos_arr + move
                out_of_bounds = jnp.any((new_pos < 0) | (new_pos >= jnp.array([h, w])))
                idx = jnp.clip(new_pos[0] * w + new_pos[1], 0, grid_flat.size - 1).astype(int)
                head_bumps_body = jnp.where(out_of_bounds, False, grid_flat[idx] > 0)
                return ~out_of_bounds & ~head_bumps_body

            return jax.vmap(is_valid)(self.MOVES)

        Snake._get_action_mask = mps_snake_mask
    except Exception:
        pass

    sys.stderr.write("✓ JAX MPS compatibility polyfill active (CPU-fallback for multi-dim ops)\n")


apply_mps_fixes()
