"""Pareto-style aggregation of returns vs proposal time across problems."""

import numpy as np

from analysis.plotting_2_trace import (
    cum_dt_prop_from_dt_prop_traces,
    load_rl_traces,
    mean_final_by_optimizer,
)
from analysis.plotting_2_util import infer_params_from_configs


def compute_pareto_data(
    results_path: str,
    exp_dir: dict[str, str],
    opt_names: list[str],
    baseline_opt: str = "turbo-one",
    exclude_opts: list[str] | None = None,
    mode: str = "both",
):
    if exclude_opts is None:
        exclude_opts = ["random"]
    opt_names_filtered = [o for o in opt_names if o not in exclude_opts]

    all_returns, all_times = _collect_returns_times(results_path, exp_dir, opt_names_filtered, mode)
    return _normalize_returns_times(all_returns, all_times, opt_names_filtered, baseline_opt)


def _collect_returns_times(results_path, exp_dir, opt_names_filtered, mode):
    all_returns, all_times = {}, {}
    for problem, exp in exp_dir.items():
        problem_seq, problem_batch = problem, f"{problem}:fn"
        inferred = infer_params_from_configs(
            results_path,
            exp,
            problem_seq=problem_seq,
            problem_batch=problem_batch,
            opt_names=opt_names_filtered,
        )
        num_reps = inferred.get("num_reps")
        num_rounds_seq, num_rounds_batch = (
            inferred.get("num_rounds_seq", 100),
            inferred.get("num_rounds_batch", 30),
        )
        num_arms_seq, num_arms_batch = (
            inferred.get("num_arms_seq", 1),
            inferred.get("num_arms_batch", 50),
        )

        configs_to_load = []
        if mode in ("seq", "both"):
            configs_to_load.append((f"{problem}_seq", problem_seq, num_arms_seq, num_rounds_seq))
        if mode in ("batch", "both"):
            configs_to_load.append((f"{problem}_batch", problem_batch, num_arms_batch, num_rounds_batch))

        for label, prob, num_arms, num_rounds in configs_to_load:
            _try_update_returns_times(
                results_path,
                exp,
                opt_names_filtered,
                num_arms=num_arms,
                num_rounds=num_rounds,
                num_reps=num_reps,
                problem=prob,
                label=label,
                all_returns=all_returns,
                all_times=all_times,
            )
    return all_returns, all_times


def _try_update_returns_times(
    results_path,
    exp,
    opt_names_filtered,
    *,
    num_arms,
    num_rounds,
    num_reps,
    problem,
    label,
    all_returns,
    all_times,
):
    try:
        data_locator, traces = load_rl_traces(
            results_path,
            exp,
            opt_names_filtered,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
        )
        all_returns[label] = mean_final_by_optimizer(data_locator, traces)
    except ValueError:
        pass
    try:
        data_locator_dt, traces_dt = load_rl_traces(
            results_path,
            exp,
            opt_names_filtered,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
            key="dt_prop",
        )
        all_times[label] = mean_final_by_optimizer(data_locator_dt, cum_dt_prop_from_dt_prop_traces(traces_dt))
    except ValueError:
        pass


def _normalize_returns_times(all_returns, all_times, opt_names_filtered, baseline_opt):
    problems = sorted(set(all_returns.keys()) & set(all_times.keys()))
    opts_in_data = set()
    for p in problems:
        opts_in_data.update(all_returns[p].keys())
        opts_in_data.update(all_times[p].keys())
    opts_in_data = [o for o in opt_names_filtered if o in opts_in_data]

    r_matrix = np.full((len(problems), len(opts_in_data)), np.nan)
    t_matrix = np.full((len(problems), len(opts_in_data)), np.nan)

    for i_p, p in enumerate(problems):
        for i_o, o in enumerate(opts_in_data):
            if o in all_returns.get(p, {}):
                r_matrix[i_p, i_o] = all_returns[p][o]
            if o in all_times.get(p, {}):
                t_matrix[i_p, i_o] = all_times[p][o]

    if baseline_opt not in opts_in_data:
        raise ValueError(f"Baseline optimizer {baseline_opt!r} not found in data. Available: {opts_in_data}")
    i_baseline = opts_in_data.index(baseline_opt)

    r_baseline = r_matrix[:, i_baseline : i_baseline + 1]
    r_centered = r_matrix - r_baseline
    rms = np.sqrt(np.nanmean(r_centered**2, axis=1, keepdims=True))
    rms = np.where(rms == 0, 1.0, rms)
    r_normalized = r_centered / rms

    t_baseline = t_matrix[:, i_baseline : i_baseline + 1]
    speedup = t_baseline / t_matrix

    return {
        "problems": problems,
        "opt_names": opts_in_data,
        "r_normalized": r_normalized,
        "speedup": speedup,
        "r_raw": r_matrix,
        "t_raw": t_matrix,
        "baseline_opt": baseline_opt,
    }
