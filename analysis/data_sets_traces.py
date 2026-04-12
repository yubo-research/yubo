import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data_io import data_is_done, read_trace_jsonl
from .data_sets_config import CACHE_DEBUG
from .data_sets_kv import load_kv


def _load_traces_jsonl_uncached(trace_dir, key="rreturn"):
    traces = []
    i_missing = []
    width = None
    trace_dir = Path(trace_dir)

    traces_subdir = trace_dir / "traces"
    if traces_subdir.exists():
        trace_dir = traces_subdir

    for fn in sorted(trace_dir.iterdir()):
        if fn.suffix != ".jsonl":
            continue
        if not data_is_done(str(fn)):
            print("NOT_DONE:", fn)
            i_missing.append(len(traces))
            traces.append(None)
            continue
        try:
            records = read_trace_jsonl(str(fn))
        except FileNotFoundError:
            raise FileNotFoundError(fn)

        trace = np.array([getattr(r, key) for r in records])
        if width is not None and len(trace) != width:
            print(f"WARNING: trace length mismatch {len(trace)} != {width} in {fn}")
        width = len(trace) if width is None else width
        traces.append(trace)

    for i in i_missing:
        traces[i] = np.nan * np.ones(width) if width else np.array([np.nan])

    traces = np.array(traces)
    return traces


@lru_cache(maxsize=1024)
def _load_traces_jsonl_cached(trace_dir, key):
    return _load_traces_jsonl_uncached(trace_dir, key=key)


def _trace_dir_cacheable_jsonl(trace_dir: str) -> bool:
    trace_dir_path = Path(trace_dir)
    traces_subdir = trace_dir_path / "traces"
    if traces_subdir.exists():
        trace_dir_path = traces_subdir
    jsonl_files = sorted(trace_dir_path.glob("*.jsonl"))
    if not jsonl_files:
        return False
    return all(data_is_done(str(fn)) for fn in jsonl_files)


def load_traces_jsonl(trace_dir, key="rreturn"):
    if not _trace_dir_cacheable_jsonl(trace_dir):
        if CACHE_DEBUG:
            print(f"CACHE BYPASS: load_traces_jsonl({str(trace_dir)[-40:]}, {key}) (incomplete)")
        return _load_traces_jsonl_uncached(trace_dir, key=key)

    if CACHE_DEBUG:
        info_before = _load_traces_jsonl_cached.cache_info()
    result = _load_traces_jsonl_cached(trace_dir, key)
    if CACHE_DEBUG:
        info_after = _load_traces_jsonl_cached.cache_info()
        if info_after.hits > info_before.hits:
            print(f"CACHE HIT: load_traces_jsonl({str(trace_dir)[-40:]}, {key})")
        else:
            print(f"CACHE MISS: load_traces_jsonl({str(trace_dir)[-40:]}, {key})")
    return result


def _load_traces_uncached(trace_dir, key, grep_for):
    trace_dir_path = Path(trace_dir)
    traces_subdir = trace_dir_path / "traces"
    if traces_subdir.exists():
        jsonl_files = list(traces_subdir.glob("*.jsonl"))
        if jsonl_files:
            key_map = {"return": "rreturn"}
            return load_traces_jsonl(trace_dir, key=key_map.get(key, key))

    traces = []
    i_missing = []
    width = None
    for fn in sorted(os.listdir(trace_dir)):
        fn = f"{trace_dir}/{fn}"
        if not os.path.isfile(fn):
            continue
        if fn.endswith(".done") or fn.endswith(".jsonl"):
            continue
        if not data_is_done(fn):
            print("NOT_DONE:", fn)
            i_missing.append(len(traces))
            traces.append(None)
            continue
        try:
            o = load_kv(fn, ["i_iter", key], grep_for=f"{grep_for}:")
        except FileNotFoundError:
            raise FileNotFoundError(fn)
        trace = o[key]
        assert width is None or len(trace) == width, (width, len(trace))
        width = len(trace)
        traces.append(trace)

    for i in i_missing:
        traces[i] = np.nan * np.ones(width)

    traces = np.array(traces)
    return traces


@lru_cache(maxsize=1024)
def _load_traces_cached(trace_dir, key, grep_for):
    return _load_traces_uncached(trace_dir, key, grep_for)


def _trace_dir_cacheable_kv(trace_dir: str) -> bool:
    trace_dir_path = Path(trace_dir)
    traces_subdir = trace_dir_path / "traces"
    if traces_subdir.exists():
        trace_dir_path = traces_subdir
    any_trace_file = False
    for fn in sorted(os.listdir(str(trace_dir_path))):
        full = str(trace_dir_path / fn)
        if not os.path.isfile(full):
            continue
        if full.endswith(".done") or full.endswith(".jsonl"):
            continue
        any_trace_file = True
        if not data_is_done(full):
            return False
    return any_trace_file


def load_traces(trace_dir, key="return", grep_for="TRACE"):
    trace_dir_path = Path(trace_dir)
    traces_subdir = trace_dir_path / "traces"
    if traces_subdir.exists():
        jsonl_files = list(traces_subdir.glob("*.jsonl"))
        if jsonl_files:
            cacheable = _trace_dir_cacheable_jsonl(trace_dir)
        else:
            cacheable = _trace_dir_cacheable_kv(trace_dir)
    else:
        cacheable = _trace_dir_cacheable_kv(trace_dir)

    if not cacheable:
        if CACHE_DEBUG:
            print(f"CACHE BYPASS: load_traces({str(trace_dir)[-40:]}, {key}) (incomplete)")
        return _load_traces_uncached(trace_dir, key, grep_for)

    if CACHE_DEBUG:
        info_before = _load_traces_cached.cache_info()
    result = _load_traces_cached(trace_dir, key, grep_for)
    if CACHE_DEBUG:
        info_after = _load_traces_cached.cache_info()
        if info_after.hits > info_before.hits:
            print(f"CACHE HIT: load_traces({trace_dir[-40:]}, {key})")
        else:
            print(f"CACHE MISS: load_traces({trace_dir[-40:]}, {key})")
    return result
