from __future__ import annotations

from typing import Any


def build_iter_record(
    *,
    algo_name: str,
    iteration: int,
    frames_per_iter: int,
    elapsed: float,
    iter_dt: float,
    metrics: dict[str, float],
    ret_best: float,
    ret_eval: float | None = None,
    eval_dt: float | None = None,
) -> dict[str, Any]:
    """Build one MJX training iteration record (jsonl + ITER: console line)."""
    step = int(iteration) * int(frames_per_iter)
    fps = float(frames_per_iter) / float(iter_dt) if iter_dt > 0 else float("nan")
    record: dict[str, Any] = {
        "iter": int(iteration),
        "step": step,
        "frames_per_iter": int(frames_per_iter),
        "elapsed": float(elapsed),
        "iter_dt": float(iter_dt),
        "fps": fps,
        "ret_rollout": float(metrics["rollout_return"]),
        "ep_ret": float(metrics["ep_ret"]),
        "ep_len": float(metrics["ep_len"]),
        "ret_best": float(ret_best),
        "rew": float(metrics["rollout_reward"]),
        "done_frac": float(metrics["done_fraction"]),
    }
    if ret_eval is not None:
        record["ret_eval"] = float(ret_eval)
    if eval_dt is not None:
        record["eval_dt"] = float(eval_dt)

    algo = str(algo_name).strip().lower()
    if algo == "ppo":
        record.update(
            {
                "loss": float(metrics["loss"]),
                "loss_pi": float(metrics["loss_objective"]),
                "loss_v": float(metrics["loss_critic"]),
                "entropy": float(metrics["entropy"]),
                "kl": float(metrics["approx_kl"]),
                "clipfrac": float(metrics["clipfrac"]),
            }
        )
        return record
    if algo == "sac":
        record.update(
            {
                "actor": float(metrics["loss_actor"]),
                "critic": float(metrics["loss_critic"]),
                "alpha": float(metrics["alpha_value"]),
                "alpha_loss": float(metrics["loss_alpha"]),
            }
        )
        return record
    raise ValueError(f"unsupported MJX algo for metrics: {algo_name!r}")
