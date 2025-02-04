import os
from pathlib import Path


def ensure_parent(path):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
