import math
import sys
import time
from contextlib import contextmanager

from analysis.data_io import TraceRecord
from common.collector import Collector
from common.experiment_seeds import global_seed_for_run
from experiments import experiment_sampler_shim as shim
from experiments.bo_console import BOConsoleCollector, print_bo_footer
from experiments.experiment_sampler_types import RunConfig, _load_attr, _SampleResult


@contextmanager
def _temporary_default_device(runtime_device: str):
    torch = shim.torch_module()
    if str(runtime_device).strip().lower() != "cuda" or not torch.cuda.is_available():
        yield
        return
    prev_default_device = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(prev_default_device)


def _collect_trace_records(
    opt,
    opt_name,
    max_iterations,
    max_total_timesteps,
    max_proposal_seconds,
    deadline,
    env_conf,
    b_trace,
):
    trace_records = []
    collector_trace = Collector()
    for i_iter, te in enumerate(
        opt.collect_trace(
            designer_name=opt_name,
            max_iterations=max_iterations,
            max_proposal_seconds=max_proposal_seconds,
            deadline=deadline,
            max_total_timesteps=max_total_timesteps,
        )
    ):
        trace_records.append(
            TraceRecord(
                i_iter=i_iter,
                dt_prop=te.dt_prop,
                dt_eval=te.dt_eval,
                rreturn=te.rreturn,
                env_name=env_conf.env_name,
                opt_name=opt_name,
                env_steps_iter=int(getattr(te, "env_steps_iter", 0)),
                env_steps_total=int(getattr(te, "env_steps_total", 0)),
            )
        )
        if b_trace:
            collector_trace(
                f"TRACE: env={env_conf.env_name} opt_name={opt_name} iter={i_iter} "
                f"proposal_dt={te.dt_prop:.3e}s eval_dt={te.dt_eval:.3e}s return={te.rreturn:.3e} "
                f"env_steps_iter={int(getattr(te, 'env_steps_iter', 0))} env_steps_total={int(getattr(te, 'env_steps_total', 0))}"
            )
    collector_trace("DONE")

    def _numeric_or(obj, name, default, cast):
        v = getattr(obj, name, default)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return cast(v)
        return default

    i_done = _numeric_or(opt, "_i_iter", 0, int)
    cum_prop = _numeric_or(opt, "_cum_dt_proposing", 0.0, float)

    stop_reason = "completed"
    if deadline is not None and time.time() >= deadline - 1.0:
        # 1-second buffer accounts for timing jitter between last iteration check and here
        stop_reason = "deadline"
    else:
        hit_round_cap = max_iterations < sys.maxsize and i_done >= int(max_iterations)
        if not hit_round_cap and (not math.isinf(float(max_proposal_seconds))) and cum_prop >= float(max_proposal_seconds) - 1e-9:
            stop_reason = "max_proposal_seconds"

    return trace_records, collector_trace, stop_reason


def _render_sample_video(
    opt,
    run_config,
    env_conf,
    video_prefix,
    video_num_episodes,
    video_num_video_episodes,
    video_episode_selection,
    video_seed_base,
):
    from pathlib import Path

    render_policy_videos_bo = _load_attr(("common", "video"), "render_policy_videos_bo")

    video_dir = Path(run_config.trace_fn).parent / "videos"
    seed_base = int(video_seed_base) if video_seed_base is not None else int(env_conf.problem_seed)
    render_policy_videos_bo(
        env_conf,
        opt.best_policy.clone(),
        video_dir=video_dir,
        video_prefix=str(video_prefix),
        num_episodes=int(video_num_episodes),
        num_video_episodes=int(video_num_video_episodes),
        episode_selection=str(video_episode_selection),
        seed_base=int(seed_base),
    )


def sample_1(run_config: RunConfig):
    import numpy as np

    rc = run_config
    problem = getattr(rc, "problem", None)
    if problem is not None:
        env_runtime = problem.env
        build_policy = problem.build_policy
        policy_tag = getattr(problem, "policy_tag", None)
    else:
        env_runtime = rc.env_conf
        if env_runtime is None:
            raise ValueError("RunConfig requires either 'problem' or 'env_conf'.")
        default_policy = _load_attr(("problems", "env_conf"), "default_policy")

        def build_policy():
            return default_policy(env_runtime)

        policy_tag = None

    max_proposal_seconds = rc.max_proposal_seconds
    if max_proposal_seconds is None:
        max_proposal_seconds = np.inf

    shim.seed_all(global_seed_for_run(env_runtime.problem_seed))

    with _temporary_default_device(getattr(rc, "runtime_device", "auto")):
        policy = build_policy()

        collector_log = BOConsoleCollector() if getattr(rc, "bo_console", True) else Collector()
        Optimizer = _load_attr(("optimizer", "optimizer"), "Optimizer")
        opt = Optimizer(
            collector_log,
            env_conf=env_runtime,
            policy_tag=policy_tag,
            policy=policy,
            num_arms=rc.num_arms,
            num_denoise_measurement=rc.num_denoise,
            num_denoise_passive=rc.num_denoise_passive,
            opt_name=rc.opt_name,
            num_rounds=rc.num_rounds,
            total_timesteps=rc.total_timesteps,
        )

        max_iterations = int(rc.num_rounds) if rc.num_rounds is not None else sys.maxsize
        trace_records, collector_trace, stop_reason = _collect_trace_records(
            opt,
            rc.opt_name,
            max_iterations,
            rc.total_timesteps,
            max_proposal_seconds,
            rc.deadline,
            env_runtime,
            rc.b_trace,
        )

        if getattr(rc, "bo_console", True):
            t0 = getattr(opt, "_t_0", None)
            if t0 is not None and isinstance(t0, (int, float)):
                total_time = time.time() - float(t0)
            else:
                total_time = 0.0
            best = getattr(opt, "r_best_est", None)
            best_val = float(best) if best is not None and isinstance(best, (int, float)) else 0.0
            print_bo_footer(best_val, max(0.0, total_time))

        if rc.video_enable and rc.video_num_video_episodes > 0 and opt.best_policy is not None:
            _render_sample_video(
                opt,
                rc,
                env_runtime,
                rc.video_prefix,
                rc.video_num_episodes,
                rc.video_num_video_episodes,
                rc.video_episode_selection,
                rc.video_seed_base,
            )

        return _SampleResult(
            collector_log=collector_log,
            collector_trace=collector_trace,
            trace_records=trace_records,
            stop_reason=stop_reason,
        )
