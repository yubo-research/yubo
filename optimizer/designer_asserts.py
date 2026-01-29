import numpy as np


def _is_scalar_return(y) -> bool:
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    a = np.asarray(y)
    return a.ndim == 0 or a.size == 1


def assert_scalar_rreturn(data) -> None:
    for d in data:
        y = d.trajectory.rreturn
        if not _is_scalar_return(y):
            a = np.asarray(y.detach().cpu().numpy() if hasattr(y, "detach") else y)
            raise AssertionError(
                f"Multi-metric rreturn is not supported (got shape={a.shape})."
            )
