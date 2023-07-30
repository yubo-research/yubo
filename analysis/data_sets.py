import numpy as np


def _extractKV(line):
    x = line.strip().split()
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
            d = _extractKV(line)
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
            d = _extractKV(line)
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


def load_traces(fn, key="return"):
    o = load_kv(fn, ["i_sample", key], grep_for="TRACE:")
    traces = []
    n_x = None
    for i_sample in np.unique(o["i_sample"]):
        i = np.where(o["i_sample"] == i_sample)[0]
        x = list(o[key][i])
        if n_x is not None:
            while len(x) < n_x:
                x.append(x[-1])
        n_x = len(x)
        traces.append(x)
    traces = np.array(traces)
    return traces


def summarize_traces(traces):
    """
    One problem, one optimizer, multiple traces
    """
    mu = traces.mean(axis=0)
    sd = traces.std(axis=0)
    se = sd / np.sqrt(len(traces))
    return mu, se


def normalize_summaries(summaries: dict):
    """
    One problem, multiple optimizers

    summaries[optimizer_name] = (mu, se)

    """
    all_mu = np.array([s[0] for s in summaries.values()])
    mean = all_mu.mean()
    std = all_mu.std()
    return {optimizer_name: ((mu - mean) / std, se / std) for optimizer_name, (mu, se) in summaries.items()}


def aggregate_normalized_summaries(normalized_summaries: dict):
    """
    Aggregate over problems.
    normalized_summaries[problem_name][optimizer_name] = (normed_mu, normed_se)
    """
    agg_by_opt = {}
    for problem_name, ns_by_opt in normalized_summaries.items():
        for optimizer_name, (mu, se) in ns_by_opt.items():
            if optimizer_name not in agg_by_opt:
                agg_by_opt[optimizer_name] = []
            agg_by_opt[optimizer_name].append((mu, se))

    summarized_agg_by_opt = {}
    for optimizer_name, agg in agg_by_opt.items():
        agg = np.array(agg)
        mu = agg[:, 0].mean(axis=0)
        se = np.sqrt((agg[:, 1] ** 2).mean(axis=0))
        summarized_agg_by_opt[optimizer_name] = (mu, se)
    return summarized_agg_by_opt


def load_as_normalized_summaries(exp_tag, problem_names, optimizer_names, data_locator):
    normalized_summaries = {}
    for problem_name in problem_names:
        summaries = {}
        for optimizer_name in optimizer_names:
            fn = data_locator(exp_tag, problem_name, optimizer_name)
            summaries[optimizer_name] = summarize_traces(load_traces(fn))
        normalized_summaries[problem_name] = normalize_summaries(summaries)
    return normalized_summaries
