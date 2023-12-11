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


def data_is_done(fn, quiet=True):
    if not os.path.exists(fn):
        return False
    with open(fn, "rb") as f:
        try:
            f.seek(-5, 2)
        except OSError:
            return False
        x = f.read(5)
        if x != b"DONE\n":
            return False
        f.seek(0, 0)
        if not quiet:
            for line in f:
                print(line.strip().decode())
        return True
