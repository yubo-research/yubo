from __future__ import annotations

# conftest.py is automatically discovered by pytest.
# We import sitecustomize to apply repository-wide compatibility patches
# (e.g. for NumPy and GPyTorch) during test runs.
import sitecustomize  # noqa: F401
