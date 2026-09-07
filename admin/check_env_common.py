from __future__ import annotations

import sys


def check_numpy_numba() -> None:
    import numba
    import numpy as np

    major, minor = (int(x) for x in np.__version__.split(".")[:2])
    if (major, minor) < (2, 3):
        raise SystemExit(f"numpy {np.__version__} too old; need >= 2.3")
    print("numpy", np.__version__, "numba", numba.__version__)


def check_core(*, hint: str) -> None:
    mac = sys.platform == "darwin"
    try:
        import faiss  # noqa: F401
        import hyperscalees  # noqa: F401
        from enn.enn.enn_class import EpistemicNearestNeighbors

        if not mac:
            import kinetix.environment  # noqa: F401
        import LassoBench  # noqa: F401
        import torch
        from pyvecch.input_transforms import Identity
    except ImportError as exc:
        raise SystemExit(f"{exc}\n{hint}") from exc

    print(
        "core ok",
        EpistemicNearestNeighbors.__name__,
        Identity.__name__,
        "torch",
        torch.__version__,
    )
