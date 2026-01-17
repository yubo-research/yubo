import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .data_io import data_is_done, read_trace_jsonl

CACHE_DEBUG = True


def clear_cache():
    _load_kv_cached.cache_clear()
    load_traces_jsonl.cache_clear()
    _load_traces_cached.cache_clear()


def cache_stats():
    return {
        "load_kv": _load_kv_cached.cache_info(),
        "load_traces_jsonl": load_traces_jsonl.cache_info(),
        "load_traces": _load_traces_cached.cache_info(),
    }


def problems_in(results_path, exp_tag):
    return sorted(os.listdir(f"{results_path}/{exp_tag}"))


def optimizers_in(results_path, exp_tag, problem):
    return sorted(os.listdir(f"{results_path}/{exp_tag}/{problem}"))


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


@lru_cache(maxsize=1024)
def _load_kv_cached(fn, keys_tuple, grep_for):
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


def load_kv(fn, keys, grep_for=None):
    if isinstance(keys, str):
        keys = keys.split(",")
    return _load_kv_cached(fn, tuple(sorted(keys)), grep_for)


@lru_cache(maxsize=1024)
def load_traces_jsonl(trace_dir, key="rreturn"):
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
def _load_traces_cached(trace_dir, key, grep_for):
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


def load_traces(trace_dir, key="return", grep_for="TRACE"):
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


def load_multiple_traces(data_locator):
    import numpy.ma as npma

    """
    Returns traces[i_problem, i_opt, i_replication, i_round]
    """

    num_bad = 0
    num_tot = 0

    problems = data_locator.problems()
    opt_names = data_locator.optimizers()

    def _report_bad(problem_name, opt_name, msg):
        nonlocal num_bad
        print("BAD:", msg, data_locator, problem_name, opt_name)
        num_bad += 1

    def _init(trace):
        if len(trace.shape) < 2:
            return None
        # Ensure float dtype from the start
        arr = np.nan * np.ones(shape=(len(problems), len(opt_names), trace.shape[0], trace.shape[1]), dtype=float)
        return arr

    traces = None
    for i_problem, problem_name in enumerate(problems):
        for i_opt, opt_name in enumerate(opt_names):
            num_tot += 1
            trace_path = data_locator(problem_name, opt_name)
            if len(trace_path) == 0:
                _report_bad(problem_name, opt_name, f"Missing data for {problem_name} {opt_name}")
                continue
            if len(trace_path) > 1:
                _report_bad(problem_name, opt_name, f"Extra data for {problem_name} {opt_name} len = {len(trace_path)}")
                continue
            trace_path = trace_path[0]

            try:
                trace = load_traces(trace_path, key=data_locator.key, grep_for=data_locator.grep_for)
            except FileNotFoundError as e:
                _report_bad(problem_name, opt_name, f"{trace_path} {repr(e)}")
                continue
            if trace is None:
                _report_bad(problem_name, opt_name, "No trace")
                continue
            if len(trace.shape) < 2:
                _report_bad(problem_name, opt_name, "Empty trace")
                continue

            if traces is None:
                traces = _init(trace)
                if traces is None:
                    _report_bad(problem_name, opt_name, "Empty trace (B)")
                    continue

            if trace.shape[0] > traces.shape[2]:
                traces_new = _init(trace)
                traces_new[:, :, : traces.shape[2], :] = traces
                traces = traces_new

            if trace.shape != traces[i_problem, i_opt, ...].shape:
                _report_bad(problem_name, opt_name, f"Warning: Trace is wrong shape {trace.shape} != {traces[i_problem, i_opt, ...].shape}")
                # continue
            # Ensure trace is numeric before assignment - force conversion to float
            if trace.dtype != np.float64 and trace.dtype != np.float32:
                try:
                    trace = np.asarray(trace, dtype=float)
                except (ValueError, TypeError):
                    # If conversion fails, try to convert element by element
                    trace = np.array([float(x) if x is not None else np.nan for x in trace.flat]).reshape(trace.shape)
            traces[i_problem, i_opt, : trace.shape[0], : trace.shape[1]] = trace

    # Ensure traces is initialized and has numeric dtype
    if traces is None:
        # No valid traces found, create empty array with float dtype
        traces = np.array([], dtype=float).reshape((len(problems), len(opt_names), 0, 0))
    else:
        # Force conversion to float - handle object arrays and other non-numeric types
        if traces.dtype == object or not np.issubdtype(traces.dtype, np.number):
            # Try to convert to float, replacing invalid values with NaN
            try:
                # First try direct conversion
                traces = np.asarray(traces, dtype=float, casting="unsafe")
            except (ValueError, TypeError):
                # If that fails, use a safer conversion that handles mixed types
                traces_flat = traces.flatten()
                traces_converted = []
                for val in traces_flat:
                    try:
                        traces_converted.append(float(val))
                    except (ValueError, TypeError):
                        traces_converted.append(np.nan)
                traces = np.array(traces_converted, dtype=float).reshape(traces.shape)

    try:
        traces = npma.masked_invalid(traces)
    except Exception as e:
        print("TP:", problems, opt_names, data_locator)
        print("TP:", traces)
        print("TP: traces dtype:", traces.dtype if traces is not None else None)
        print("TP: traces shape:", traces.shape if traces is not None else None)
        raise e
    if num_bad > 0:
        print(f"\n{num_bad} / {num_tot} files bad. {100 * traces.mask.mean():.1f}% missing data")
    else:
        print("No bad data")
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
