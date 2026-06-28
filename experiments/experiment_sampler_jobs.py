import os

from analysis.data_io import write_config
from common.experiment_seeds import (
    noise_seed_0_from_problem_seed,
    problem_seed_from_rep_index,
)
from experiments import experiment_sampler_shim as shim
from experiments.experiment_sampler_dispatch import scan_local
from experiments.experiment_sampler_types import ExperimentConfig, RunConfig
from problems.eggroll_jax_flags import eggroll_jax_sim_enabled


def mk_replicates(config: ExperimentConfig) -> list[RunConfig]:
    out_dir = config.to_dir_name()

    os.makedirs(out_dir, exist_ok=True)
    write_config(out_dir, config.to_dict())
    print(f"PARAMS: {config}")
    run_configs = []
    for i_rep in range(config.num_reps):
        trace_fn = f"{out_dir}/traces/{i_rep:05d}"
        jsonl_fn = trace_fn + ".jsonl"
        if shim.data_is_done(trace_fn) or shim.data_is_done(jsonl_fn):
            print(f"Skipping trace_fn = {trace_fn}. Already done.")
            continue
        else:
            problem_seed = problem_seed_from_rep_index(i_rep)
            initial_policy_checkpoint = _format_initial_policy_checkpoint(
                config.initial_policy_checkpoint,
                rep_index=i_rep,
                problem_seed=problem_seed,
            )
            problem = shim.build_problem(
                config.env_tag,
                config.policy_tag,
                problem_seed=problem_seed,
                noise_seed_0=noise_seed_0_from_problem_seed(problem_seed),
                isaaclab_video=bool(config.video_enable and config.video_num_video_episodes > 0),
            )
            if eggroll_jax_sim_enabled(config.opt_name):
                setattr(problem.env, "eggroll_jax_sim", True)
                setattr(problem.env, "eggroll_population", int(config.num_arms))
                setattr(problem.env, "runtime_device", str(config.runtime_device))
            run_configs.append(
                RunConfig(
                    trace_fn=trace_fn,
                    problem=problem,
                    opt_name=config.opt_name,
                    num_rounds=config.num_rounds,
                    total_timesteps=config.total_timesteps,
                    num_arms=config.num_arms,
                    num_denoise=config.num_denoise,
                    num_denoise_passive=config.num_denoise_passive,
                    max_proposal_seconds=config.max_proposal_seconds,
                    b_trace=config.b_trace,
                    video_enable=config.video_enable,
                    video_num_episodes=config.video_num_episodes,
                    video_num_video_episodes=config.video_num_video_episodes,
                    video_episode_selection=config.video_episode_selection,
                    video_seed_base=config.video_seed_base,
                    video_prefix=config.video_prefix,
                    runtime_device=config.runtime_device,
                    initial_policy_checkpoint=initial_policy_checkpoint,
                )
            )
    return run_configs


def _format_initial_policy_checkpoint(
    value: str | None,
    *,
    rep_index: int,
    problem_seed: int,
) -> str | None:
    if value is None:
        return None
    return str(value).format(
        rep=rep_index,
        i_rep=rep_index,
        rep_index=rep_index,
        problem_seed=problem_seed,
    )


def count_local_trace_jobs(configs: list[ExperimentConfig]) -> tuple[int, int, int]:
    """Count local traces using the same rules as :func:`mk_replicates`.

    Returns ``(n_complete, n_remaining, n_total)`` over every replicate in ``configs``.
    """
    n_complete = 0
    n_total = 0
    for config in configs:
        out_dir = config.to_dir_name()
        for i_rep in range(int(config.num_reps)):
            n_total += 1
            trace_fn = f"{out_dir}/traces/{i_rep:05d}"
            jsonl_fn = trace_fn + ".jsonl"
            if shim.data_is_done(trace_fn) or shim.data_is_done(jsonl_fn):
                n_complete += 1
    return n_complete, n_total - n_complete, n_total


def sampler(config: ExperimentConfig, distributor_fn):
    run_configs = shim.mk_replicates(config)
    if distributor_fn is scan_local:
        distributor_fn(
            run_configs,
            max_total_seconds=config.max_total_seconds,
            local_workers=config.local_workers,
            env_tag=config.env_tag,
        )
    else:
        distributor_fn(run_configs)


def prep_args_1(
    results_dir,
    exp_dir,
    problem,
    opt,
    num_arms,
    num_replications,
    num_rounds,
    noise=None,
    num_denoise=None,
    num_denoise_passive=None,
    policy_tag="pure-function",
    initial_policy_checkpoint=None,
) -> ExperimentConfig:
    assert noise is None, "NYI"

    full_exp_dir = f"{results_dir}/{exp_dir}"

    return ExperimentConfig(
        exp_dir=full_exp_dir,
        env_tag=problem,
        opt_name=opt,
        num_arms=num_arms,
        num_reps=num_replications,
        num_rounds=num_rounds,
        total_timesteps=None,
        num_denoise=num_denoise,
        num_denoise_passive=num_denoise_passive,
        policy_tag=policy_tag,
        initial_policy_checkpoint=initial_policy_checkpoint,
    )


def prep_d_args(
    results_dir,
    exp_dir,
    funcs,
    dims,
    num_arms,
    num_replications,
    opts,
    noises,
    num_rounds=3,
    func_category="f",
    num_denoise=None,
    num_denoise_passive=None,
    policy_tag="pure-function",
) -> list[ExperimentConfig]:
    configs = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                for noise in noises:
                    problem = f"{func_category}:{func}-{dim}d"
                    configs.append(
                        prep_args_1(
                            results_dir,
                            exp_dir,
                            problem,
                            opt,
                            num_arms,
                            num_replications,
                            num_rounds,
                            noise,
                            num_denoise=num_denoise,
                            num_denoise_passive=num_denoise_passive,
                            policy_tag=policy_tag,
                        )
                    )
    return configs
