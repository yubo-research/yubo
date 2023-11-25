import os
import sys
from contextlib import contextmanager


@contextmanager
def data_writer(out_fn):
    if out_fn != "-":
        f = open(out_fn, "w")
    else:
        f = sys.stdout
    try:
        yield f
    finally:
        if f != sys.stdout:
            f.close()


def data_is_done(fn):
    if not os.path.exists(fn):
        return False
    with open(fn, "rb") as f:
        try:
            f.seek(-5, 2)
        except OSError:
            return False
        x = f.read(5)
        return x == b"DONE\n"
