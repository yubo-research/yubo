import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .data_io import data_is_done, read_trace_jsonl

CACHE_DEBUG = False


def clear_cache():
    _load_kv_cached.cache_clear()
    _load_traces_jsonl_cached.cache_clear()
    _load_traces_cached.cache_clear()


def cache_stats():
    return {
        "load_kv": _load_kv_cached.cache_info(),
        "load_traces_jsonl": _load_traces_jsonl_cached.cache_info(),
        "load_traces": _load_traces_cached.cache_info(),
    }


def problems_in(results_path, exp_tag):
    return sorted(os.listdir(f"{results_path}/{exp_tag}"))


def _optimizers_in(results_path, exp_tag, problem):
    return sorted(os.listdir(f"{results_path}/{exp_tag}/{problem}"))


optimizers_in = _optimizers_in


def all_in(results_path, exp_tag):
    optimizers = set()
    problems = problems_in(results_path, exp_tag)
    for problem in problems:
        for optimizer in optimizers_in(results_path, exp_tag, problem):
            optimizers.update(optimizers_in(results_path, exp_tag, problem))
    return problems, sorted(optimizers)


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


def _ensure_float_trace(trace):
    if trace.dtype == np.float64 or trace.dtype == np.float32:
        return trace
    try:
        return np.asarray(trace, dtype=float)
    except (ValueError, TypeError):
        return np.array([float(x) if x is not None else np.nan for x in trace.flat]).reshape(trace.shape)


def _ensure_float_traces(traces):
    if traces.dtype != object and np.issubdtype(traces.dtype, np.number):
        return traces
    try:
        return np.asarray(traces, dtype=float, casting="unsafe")
    except (ValueError, TypeError):

        def _safe_float(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan

        return np.array([_safe_float(v) for v in traces.flatten()], dtype=float).reshape(traces.shape)


def _validate_trace_path(trace_path, problem_name, opt_name, report_bad):
    if len(trace_path) == 0:
        report_bad(problem_name, opt_name, f"Missing data for {problem_name} {opt_name}")
        return None
    if len(trace_path) > 1:
        report_bad(problem_name, opt_name, f"Extra data len = {len(trace_path)}")
        return None
    return trace_path[0]


def _load_and_validate_trace(trace_path, data_locator, problem_name, opt_name, report_bad):
    try:
        trace = load_traces(trace_path, key=data_locator.key, grep_for=data_locator.grep_for)
    except FileNotFoundError as e:
        report_bad(problem_name, opt_name, f"{trace_path} {repr(e)}")
        return None
    if trace is None or len(trace.shape) < 2:
        report_bad(problem_name, opt_name, "No/empty trace")
        return None
    return trace


def load_multiple_traces(data_locator):
    import numpy.ma as npma

    num_bad = 0
    num_tot = 0
    problems = data_locator.problems()
    opt_names = data_locator.optimizers()

    def _report_bad(problem_name, opt_name, msg):
        nonlocal num_bad
        print("BAD:", msg, data_locator, problem_name, opt_name)
        num_bad += 1

    def _init_traces(trace):
        return (
            np.nan
            * np.ones(
                (len(problems), len(opt_names), trace.shape[0], trace.shape[1]),
                dtype=float,
            )
            if len(trace.shape) >= 2
            else None
        )

    traces = None
    for i_problem, problem_name in enumerate(problems):
        for i_opt, opt_name in enumerate(opt_names):
            num_tot += 1
            trace_path = _validate_trace_path(data_locator(problem_name, opt_name), problem_name, opt_name, _report_bad)
            if trace_path is None:
                continue
            trace = _load_and_validate_trace(trace_path, data_locator, problem_name, opt_name, _report_bad)
            if trace is None:
                continue
            if traces is None:
                traces = _init_traces(trace)
                if traces is None:
                    _report_bad(problem_name, opt_name, "Empty trace (B)")
                    continue
            if trace.shape[0] > traces.shape[2]:
                traces_new = _init_traces(trace)
                traces_new[:, :, : traces.shape[2], :] = traces
                traces = traces_new
            if trace.shape != traces[i_problem, i_opt, ...].shape:
                _report_bad(problem_name, opt_name, f"Wrong shape {trace.shape}")
            trace = _ensure_float_trace(trace)
            traces[i_problem, i_opt, : trace.shape[0], : trace.shape[1]] = trace

    if traces is None:
        traces = np.array([], dtype=float).reshape((len(problems), len(opt_names), 0, 0))
    else:
        traces = _ensure_float_traces(traces)

    traces = npma.masked_invalid(traces)
    if num_bad > 0:
        print(f"\n{num_bad} / {num_tot} files bad. {100 * traces.mask.mean():.1f}% missing data")
    return traces


def range_summarize(traces: np.ndarray):
    """
    traces[i_problem, i_opt, i_replication, i_round]
    Returns mu,se ~ [i_problem, i_opt]
    """
    y_min = traces.min(axis=1, keepdims=True)
    y_max = traces.max(axis=1, keepdims=True)

    numer = traces - y_min
    denom = y_max - y_min
    i_zero = np.where(denom == 0)[0]
    numer[i_zero] = 0.5
    denom[i_zero] = 1
    z = numer / denom

    # Take last round
    z = z[..., -1]

    z = z.swapaxes(0, 1)
    z = z.reshape(z.shape[0], z.shape[1] * z.shape[2])

    mu = z.mean(axis=-1)
    se = z.std(axis=-1) / np.sqrt(z.shape[-1])

    return mu, se


def rank_summarize(traces: np.array):
    """
    traces[i_problem, i_opt, i_replication, i_round]
    Returns mu,se ~ [i_problem, i_opt]
    """

    # over opt (method)

    z = rankdata(traces, axis=1, nan_policy="omit")
    z = (z - 1) / (z.shape[1] - 1)
    z = z.mean(axis=-1)

    z = z.swapaxes(0, 1)
    z = z.reshape(z.shape[0], z.shape[1] * z.shape[2])

    mu = z.mean(axis=-1)
    se = z.std(axis=-1) / np.sqrt(z.shape[-1])

    return mu, se
