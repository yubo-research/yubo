from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp


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


def _log_iteration(
    *,
    logger,
    iteration: int,
    iterations: int,
    frames_per_iter: int,
    elapsed: float,
    values: dict[str, float],
    best_return: float,
    eval_return: float | None,
    algo_name: str,
    prefix: str,
) -> None:
    if eval_return is not None:
        logger.log_eval_iteration(
            iteration,
            iterations,
            frames_per_iter,
            eval_return=eval_return,
            best_return=best_return,
            algo_metrics=values,
            elapsed=elapsed,
            algo_name=algo_name,
            prefix=prefix,
        )
        return
    logger.log_progress_iteration(
        iteration,
        iterations,
        frames_per_iter,
        elapsed=elapsed,
        best_return=best_return,
        algo_metrics=values,
        algo_name=algo_name,
        prefix=prefix,
    )


def run_mjx_training_loop(
    *,
    config,
    runtime,
    state,
    train_step,
    eval_step,
    result_fn,
    eval_args_fn,
    record_fn,
    checkpoint_fn=None,
    restore_fn=None,
    algo_name: str,
    prefix: str,
):
    from rl import logger

    frames_per_iter = int(config.collector.num_envs) * int(config.collector.num_steps)
    iterations = int(config.collector.total_frames) // frames_per_iter
    exp_dir = Path(config.exp_dir) / f"seed_{int(config.seed)}"
    metrics_path = exp_dir / "metrics.jsonl"
    ckpt_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)

    mngr = _checkpoint_manager(ckpt_dir)
    if config.checkpoint.resume_from:
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

    start = time.time()
    best_return = float("-inf")
    last_return = float("nan")
    eval_key = runtime.jax.random.key(int(config.seed) + 1000)

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
    )

    for iteration in range(1, iterations + 1):
        iter_start = time.time()
        state, metrics = train_step(state)
        values = _metrics_to_float(runtime, metrics)
        last_return = _last_rollout_return(values)
        iter_dt = time.time() - iter_start
        eval_interval = int(config.eval.interval)
        do_eval = eval_interval > 0 and (iteration == 1 or iteration % eval_interval == 0 or iteration == iterations)
        eval_return = None
        if do_eval:
            eval_key, subkey = runtime.jax.random.split(eval_key)
            eval_return = float(eval_step(*eval_args_fn(state), subkey))
            best_return = max(best_return, eval_return)
        elif _is_finite(last_return):
            best_return = max(best_return, last_return)

        should_log = iteration == 1 or iteration % int(config.log_interval) == 0 or iteration == iterations or do_eval
        if should_log:
            elapsed = time.time() - start
            record = record_fn(
                iteration=iteration,
                frames_per_iter=frames_per_iter,
                elapsed=elapsed,
                iter_dt=iter_dt,
                metrics=values,
                ret_best=best_return,
            )
            if eval_return is not None:
                record["ret_eval"] = eval_return
            _log_iteration(
                logger=logger,
                iteration=iteration,
                iterations=iterations,
                frames_per_iter=frames_per_iter,
                elapsed=elapsed,
                values=values,
                best_return=best_return,
                eval_return=eval_return,
                algo_name=algo_name,
                prefix=prefix,
            )
            logger.append_metrics(metrics_path, record)

        if config.checkpoint.interval and iteration % int(config.checkpoint.interval) == 0:
            save_items = checkpoint_fn(state) if checkpoint_fn else state
            mngr.save(iteration, save_items)
            mngr.wait_until_finished()

    save_items = checkpoint_fn(state) if checkpoint_fn else state
    mngr.save(iterations, save_items)
    mngr.wait_until_finished()
    logger.log_run_footer(best_return, iterations, time.time() - start, algo_name=algo_name)
    return result_fn(best_return, last_return, iterations, frames_per_iter)
