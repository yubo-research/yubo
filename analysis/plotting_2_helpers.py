"""Helpers for loading RL traces with optional cumulative dt_prop metadata."""

from analysis.plotting_2_trace import (
    cum_dt_prop_from_dt_prop_traces,
    load_rl_traces,
    median_final_by_optimizer,
    print_cum_dt_prop,
)
from analysis.plotting_types import RLTracesWithCumDtProp


def _load_rl_with_cum_dt_prop(results_path, exp_dir, opt_names, *, num_arms, num_rounds, num_reps, problem):
    data_locator, traces = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
    )
    cum_dt_prop = None
    try:
        data_locator_dt, traces_dt = load_rl_traces(
            results_path,
            exp_dir,
            opt_names,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
            key="dt_prop",
        )
        cum_dt_prop = median_final_by_optimizer(data_locator_dt, cum_dt_prop_from_dt_prop_traces(traces_dt))
    except ValueError:
        cum_dt_prop = None
    return RLTracesWithCumDtProp(data_locator=data_locator, traces=traces, cum_dt_prop=cum_dt_prop)


def _try_load_rl_with_cum_dt_prop(results_path, exp_dir, opt_names, *, num_arms, num_rounds, num_reps, problem):
    try:
        return _load_rl_with_cum_dt_prop(
            results_path,
            exp_dir,
            opt_names,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
        )
    except ValueError:
        return None


def _print_cum_dt_props(
    cum_dt_prop_seq,
    cum_dt_prop_batch,
    *,
    opt_names_seq,
    opt_names_batch,
    opt_names_all,
    problem_seq,
    problem_batch,
):
    print_cum_dt_prop(
        cum_dt_prop_seq,
        opt_names_all or opt_names_seq,
        header=f"cumulative proposal times ({problem_seq})",
    )
    print_cum_dt_prop(
        cum_dt_prop_batch,
        opt_names_all or opt_names_batch,
        header=f"cumulative proposal times ({problem_batch})",
    )
