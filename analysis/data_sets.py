import os

import numpy as np
from scipy.stats import rankdata

from .data_io import data_is_done


def problems_in(exp_tag):
    return sorted(os.listdir(f"/Users/dsweet2/Projects/bbo/results/{exp_tag}"))


def optimizers_in(exp_tag, problem):
    return sorted(os.listdir(f"/Users/dsweet2/Projects/bbo/results/{exp_tag}/{problem}"))


def all_in(exp_tag):
    optimizers = set()
    problems = problems_in(exp_tag)
    for problem in problems:
        for optimizer in optimizers_in(exp_tag, problem):
            optimizers.update(optimizers_in(exp_tag, problem))
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
        for line in f.readlines():
            d = extract_kv(line.strip().split())
            if skeys.issubset(set(d.keys())):
                data.append([float(d[k]) for k in keys])
    return np.array(data).squeeze()


def load_kv(fn, keys, grep_for=None):
    if isinstance(keys, str):
        keys = keys.split(",")
    skeys = set(keys)
    data = {k: [] for k in skeys}
    with open(fn) as f:
        for line in f.readlines():
            if grep_for is not None and grep_for not in line:
                continue
            i = line.find("[INFO")
            if i is not None:
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


def load_traces(trace_dir, key="return"):
    traces = []
    i_missing = []
    width = None
    for fn in sorted(os.listdir(trace_dir)):
        fn = f"{trace_dir}/{fn}"
        if not data_is_done(fn):
            print("NOT_DONE:", fn)
            i_missing.append(len(traces))
            traces.append(None)
            continue
        try:
            o = load_kv(fn, ["i_iter", key], grep_for="TRACE:")
        except FileNotFoundError:
            raise FileNotFoundError(fn)
        trace = o[key]
        assert width is None or len(trace) == width, (width, len(trace))
        width = len(trace)
        traces.append(trace)

    for i in i_missing:
        traces[i] = np.nan * np.ones(width)

    traces = np.array(traces)
    # print (f"Loaded {len(traces)} traces from {fn}")
    return traces


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
        return np.nan * np.ones(shape=(len(problems), len(opt_names), trace.shape[0], trace.shape[1]))

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
                trace = load_traces(trace_path)
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
                _report_bad(problem_name, opt_name, f"Trace is wrong shape {trace.shape} != {traces[i_problem, i_opt, ...].shape}")
                # continue
            traces[i_problem, i_opt, : trace.shape[0], : trace.shape[1]] = trace

    try:
        traces = npma.masked_invalid(traces)
    except:
        print("TP:", problems, opt_names)
        assert False
    if num_bad > 0:
        print(f"\n{num_bad} / {num_tot} files bad. {100*traces.mask.mean():.1f}% missing data")
    else:
        print("No bad data")
    return npma.masked_invalid(traces)


def range_summarize(traces: np.ndarray):
    """
    traces[i_problem, i_opt, i_replication, i_round]
    Returns mu,se ~ [i_problem, i_opt]
    """
    y_min = traces.min(axis=1, keepdims=True)
    y_max = traces.max(axis=1, keepdims=True)
    z = (traces - y_min) / (y_max - y_min)

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
    z = rankdata(traces, axis=1)
    z = (z - 1) / (z.shape[1] - 1)
    z = z.mean(axis=-1)

    z = z.swapaxes(0, 1)
    z = z.reshape(z.shape[0], z.shape[1] * z.shape[2])

    mu = z.mean(axis=-1)
    se = z.std(axis=-1) / np.sqrt(z.shape[-1])

    return mu, se
