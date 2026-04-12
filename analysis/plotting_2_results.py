"""High-level `plot_results` entrypoint for paper-style RL figure bundles."""

from analysis.plotting_types import PlotResultsResult as _PlotResultsResult

_LONG_NAMES = {
    "tlunar": "LunarLander-v3",
    "push": "Push-v3",
    "hop": "Hopper-v5",
    "bw-heur": "BipedalWalker-v3",
    "dna": "LASSO-DNA",
}


def plot_results(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    problem: str,
    num_reps: int | None = None,
    num_reps_seq: int | None = None,
    num_reps_batch: int | None = None,
    num_rounds_seq: int | None = None,
    num_rounds_batch: int | None = None,
    num_arms_seq: int | None = None,
    num_arms_batch: int | None = None,
    exclude_seq: list[str] | None = None,
    exclude_batch: list[str] | None = None,
):
    from analysis import plotting_2_comparison as cmp
    from analysis import plotting_2_trace as tr
    from analysis import plotting_2_util as u2

    problem_name = _LONG_NAMES.get(problem, problem)
    problem_seq, problem_batch = problem, f"{problem}:fn"
    opt_names_seq = [o for o in opt_names if o not in (exclude_seq or [])]
    opt_names_batch = [o for o in opt_names if o not in (exclude_batch or [])]

    inferred = u2.infer_params_from_configs(
        results_path,
        exp_dir,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        opt_names=opt_names,
    )
    inferred_num_reps = inferred.get("num_reps")
    shared_num_reps = num_reps if num_reps is not None else inferred_num_reps
    num_reps_seq = num_reps_seq if num_reps_seq is not None else shared_num_reps
    num_reps_batch = num_reps_batch if num_reps_batch is not None else shared_num_reps
    num_rounds_seq = num_rounds_seq or inferred.get("num_rounds_seq", 100)
    num_rounds_batch = num_rounds_batch or inferred.get("num_rounds_batch", 30)
    num_arms_seq = num_arms_seq or inferred.get("num_arms_seq", 1)
    num_arms_batch = num_arms_batch or inferred.get("num_arms_batch", 50)

    tr.print_dataset_summary(
        results_path,
        exp_dir,
        problem=problem_seq,
        opt_names=opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps_seq,
    )
    tr.print_dataset_summary(
        results_path,
        exp_dir,
        problem=problem_batch,
        opt_names=opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps_batch,
    )

    print(problem_name)
    fig_curves, axs_curves, seq_data, batch_data = cmp.plot_rl_comparison(
        results_path,
        exp_dir,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_reps=num_reps,
        num_reps_seq=num_reps_seq,
        num_reps_batch=num_reps_batch,
        num_rounds_seq=num_rounds_seq,
        num_rounds_batch=num_rounds_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=None,
        cum_dt_prop=problem in {"tlunar", "push"},
        opt_names_all=opt_names,
        show_titles=False,
        print_titles=True,
    )

    print(f"{problem_name} Final Performance Comparison (±2 SE)")
    fig_final, axs_final, _, _ = cmp.plot_rl_final_comparison(
        results_path,
        exp_dir,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_reps=num_reps,
        num_reps_seq=num_reps_seq,
        num_reps_batch=num_reps_batch,
        num_rounds_seq=num_rounds_seq,
        num_rounds_batch=num_rounds_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=None,
        opt_names_all=opt_names,
        show_titles=False,
        print_titles=True,
    )

    return _PlotResultsResult(
        curves=(fig_curves, axs_curves),
        final=(fig_final, axs_final),
        seq_data=seq_data,
        batch_data=batch_data,
    )
