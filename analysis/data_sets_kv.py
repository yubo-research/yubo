from functools import lru_cache

import numpy as np

from .data_io import data_is_done


def extract_kv(x):
    d = {}
    for i in range(0, len(x) - 1):
        if x[i] != "=":
            continue
        k = x[i - 1]
        v = x[i + 1]
        d[k] = v
    return d


def load(fn, keys):
    skeys = set(keys)
    data = []
    with open(fn) as f:
        for line in f:
            d = extract_kv(line.strip().split())
            if skeys.issubset(set(d.keys())):
                data.append([float(d[k]) for k in keys])
    return np.array(data).squeeze()


def _load_kv_uncached(fn, keys_tuple, grep_for):
    skeys = set(keys_tuple)
    data = {k: [] for k in skeys}
    with open(fn) as f:
        for line in f:
            if grep_for is not None and grep_for not in line:
                continue
            i = line.find("[INFO")
            if i != -1:
                line = line[:i]
            d = extract_kv(line.strip().split())
            for k in skeys:
                if k in d:
                    data[k].append(float(d[k]))
    out = {}
    for k, v in data.items():
        if len(v) > 0:
            out[k] = np.array(v).squeeze()
            if len(out[k].shape) == 0:
                out[k] = np.array([out[k]])
    return out


@lru_cache(maxsize=1024)
def _load_kv_cached(fn, keys_tuple, grep_for):
    return _load_kv_uncached(fn, keys_tuple, grep_for)


def load_kv(fn, keys, grep_for=None):
    if isinstance(keys, str):
        keys = keys.split(",")
    keys_tuple = tuple(sorted(keys))
    if data_is_done(fn):
        return _load_kv_cached(fn, keys_tuple, grep_for)
    return _load_kv_uncached(fn, keys_tuple, grep_for)
