import importlib.util
from pathlib import Path

# Smart shim that routes 'import enn' to local source or installed package
# using __path__ indirection.

# Possible locations for the local 'enn' repository source
possible_paths = [
    Path(__file__).resolve().parents[2] / "enn" / "src" / "enn",  # Local dev
    Path("/root/enn_local/src/enn"),  # Modal mount
]

real_path = None
for p in possible_paths:
    if p.exists():
        real_path = p
        break

if real_path:
    __path__ = [str(real_path)]
else:
    # Try to find installed 'ennbo' package
    spec = importlib.util.find_spec("ennbo")
    if spec:
        __path__ = spec.submodule_search_locations
    else:
        __path__ = []

if __path__:
    # Now we can import from ourselves!
    # Since __path__ is set, these will look in the real source.
    try:
        from .enn.enn_class import EpistemicNearestNeighbors
        from .enn.enn_fit import enn_fit
        from .enn.enn_params_class import ENNParams
    except ImportError:
        try:
            # Fallback for different internal structures
            from .enn import ENNParams, EpistemicNearestNeighbors, enn_fit
        except ImportError:
            pass

__all__ = ["EpistemicNearestNeighbors", "ENNParams", "enn_fit"]
