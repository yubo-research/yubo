from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

from analysis.data_io import mark_done, write_config, write_summary_json
from rl.mjx_metrics import build_iter_record


def _checkpoint_manager(ckpt_dir: Path):
    return ocp.CheckpointManager(
        ckpt_dir.absolute(),
        ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True),
    )


def _metrics_to_float(runtime, metrics) -> dict[str, float]:
    return {name: float(value) for name, value in runtime.jax.device_get(metrics).items()}


def _last_rollout_return(values: dict[str, float]) -> float:
    ep_ret = values.get("ep_ret", float("nan"))
    return ep_ret if not np.isnan(ep_ret) else values["rollout_return"]


def _is_finite(value: float) -> bool:
    return not np.isnan(value) and not np.isinf(value)


def _restore_mjx_state_if_requested(
    *,
    config,
    state,
    mngr,
    checkpoint_fn,
    restore_fn,
    prefix: str,
):
    if not config.checkpoint.resume_from:
        return state
    from rl import logger

    resume_path = Path(config.checkpoint.resume_from)
    if not resume_path.is_absolute():
        resume_path = Path.cwd() / resume_path
    restore_items = checkpoint_fn(state) if checkpoint_fn else state
    restored = mngr.restore(mngr.latest_step(), items=restore_items)
    if restore_fn:
        state = restore_fn(state, restored)
    else:
        state = restored
    logger.log_rl_status(f"{prefix}Resumed from {resume_path} at step {mngr.latest_step()}")
    return state


def run_mjx_training_loop(
    *,
    config,
    runtime,
    state,
    train_step,
    eval_step,
    result_fn,
    eval_args_fn,
    checkpoint_fn=None,
    restore_fn=None,
    algo_name: str,
    prefix: str,
):
    from rl import logger

    logger.configure_logging()

    frames_per_iter = int(config.collector.num_envs) * int(config.collector.num_steps)
    iterations = int(config.collector.total_frames) // frames_per_iter
    exp_dir = Path(config.exp_dir) / f"seed_{int(config.seed)}"
    metrics_path = exp_dir / "metrics.jsonl"
    ckpt_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())

    mngr = _checkpoint_manager(ckpt_dir)
    state = _restore_mjx_state_if_requested(
        config=config,
        state=state,
        mngr=mngr,
        checkpoint_fn=checkpoint_fn,
        restore_fn=restore_fn,
        prefix=prefix,
    )

    start = time.time()
    best_return = float("-inf")
    last_return = float("nan")
    eval_key = runtime.jax.random.key(int(config.seed) + 1000)
    stop_reason = "completed"

    logger.log_run_header_basic(
        algo_name=algo_name,
        env_tag=config.env_tag,
        seed=config.seed,
        backbone_name="mlp",
        from_pixels=False,
        obs_dim=runtime.obs_dim,
        act_dim=runtime.act_dim,
        frames_per_batch=frames_per_iter,
        num_iterations=iterations,
        device_type=str(runtime.jax.default_backend()),
        config_obj=config,
        prefix=prefix,
    )
    logger.log_rl_status(f"{prefix}metrics={metrics_path} checkpoints={ckpt_dir} frames_per_iter={frames_per_iter} iters={iterations}")

    for iteration in range(1, iterations + 1):
        iter_start = time.time()
        state, metrics = train_step(state)
        values = _metrics_to_float(runtime, metrics)
        last_return = _last_rollout_return(values)
        iter_dt = time.time() - iter_start
        eval_interval = int(config.eval.interval)
        do_eval = eval_interval > 0 and (iteration == 1 or iteration % eval_interval == 0 or iteration == iterations)
        eval_return = None
        eval_dt = None
        if do_eval:
            eval_start = time.time()
            eval_key, subkey = runtime.jax.random.split(eval_key)
            eval_return = float(eval_step(*eval_args_fn(state), subkey))
            eval_dt = time.time() - eval_start
            best_return = max(best_return, eval_return)
        elif _is_finite(last_return):
            best_return = max(best_return, last_return)

        should_log = iteration == 1 or iteration % int(config.log_interval) == 0 or iteration == iterations or do_eval
        if should_log:
            elapsed = time.time() - start
            record = build_iter_record(
                algo_name=algo_name,
                iteration=iteration,
                frames_per_iter=frames_per_iter,
                elapsed=elapsed,
                iter_dt=iter_dt,
                metrics=values,
                ret_best=best_return,
                ret_eval=eval_return,
                eval_dt=eval_dt,
            )
            logger.log_rl_iter(record, metrics_path=metrics_path)

        if config.checkpoint.interval and iteration % int(config.checkpoint.interval) == 0:
            save_items = checkpoint_fn(state) if checkpoint_fn else state
            mngr.save(iteration, save_items)
            mngr.wait_until_finished()

    return _finalize_mjx_training(
        logger=logger,
        mngr=mngr,
        state=state,
        checkpoint_fn=checkpoint_fn,
        iterations=iterations,
        metrics_path=metrics_path,
        start=start,
        stop_reason=stop_reason,
        best_return=best_return,
        last_return=last_return,
        frames_per_iter=frames_per_iter,
        algo_name=algo_name,
        result_fn=result_fn,
    )


def _finalize_mjx_training(
    *,
    logger,
    mngr,
    state,
    checkpoint_fn,
    iterations: int,
    metrics_path,
    start: float,
    stop_reason: str,
    best_return: float,
    last_return: float,
    frames_per_iter: int,
    algo_name: str,
    result_fn,
):
    wall_seconds = time.time() - start
    save_items = checkpoint_fn(state) if checkpoint_fn else state
    mngr.save(iterations, save_items)
    mngr.wait_until_finished()
    write_summary_json(str(metrics_path), wall_seconds, stop_reason)
    mark_done(str(metrics_path))
    logger.log_run_footer(best_return, iterations, wall_seconds, algo_name=algo_name)
    return result_fn(best_return, last_return, iterations, frames_per_iter)
