import numpy as np
from scipy.stats import rankdata

from .data_sets_traces import load_traces


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
    requested_num_reps = getattr(data_locator, "num_reps", None)

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
            trace_path = _validate_trace_path(
                data_locator(problem_name, opt_name),
                problem_name,
                opt_name,
                _report_bad,
            )
            if trace_path is None:
                continue
            trace = _load_and_validate_trace(trace_path, data_locator, problem_name, opt_name, _report_bad)
            if trace is None:
                continue
            if requested_num_reps is not None:
                trace = trace[:requested_num_reps, :]
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
