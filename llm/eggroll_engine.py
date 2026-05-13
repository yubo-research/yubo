from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm.config import LLMConfig
from llm.es import (
    summarize_fitness,
)
from llm.lora import LoraTemplate
from llm.tasks import MathTask


@dataclass(frozen=True)
class EggrollArgs:
    base_seed: int
    population_size: int
    num_iterations: int
    sigma: float
    learning_rate: float
    lora_r: int
    lora_alpha: int
    steps_per_adapter: int
    max_tokens: int
    temperature: float
    samples_per_prompt: int
    prompt_batch_size: int
    pass_at_k: bool
    normalize_with_std: bool
    scale_lr_in_grad: bool
    num_gpus: int
    num_engines: int
    tensor_parallel_size: int
    steps_per_eval: int
    eval_batch_size: int
    save_freq: int | None
    checkpoint_dir: str | None
    use_wandb: bool
    wandb_project: str
    wandb_name: str | None


def train_loop(
    ray: Any,
    *,
    engines: list[Any],
    args: EggrollArgs,
    template: LoraTemplate,
    task: Any,
    eval_task: MathTask | None,
    sampling_kwargs: dict[str, Any],
    eval_sampling_kwargs: dict[str, Any],
    wandb_run: Any | None,
) -> dict[str, Any]:
    loras_per_engine = args.population_size // args.num_engines
    engine_pop_indices = [list(range(i * loras_per_engine, (i + 1) * loras_per_engine)) for i in range(args.num_engines)]
    engine_paths: list[list[str]] | None = None
    last_summary = None
    start_time = time.time()

    for es_step in range(args.num_iterations):
        iter_start = time.time()
        if es_step % args.steps_per_adapter == 0 or engine_paths is None:
            engine_paths = ray.get([engines[i].generate_local_adapters.remote(engine_pop_indices[i], es_step, args) for i in range(args.num_engines)])

        eval_info = run_eval(
            ray,
            engines=engines,
            eval_task=eval_task,
            args=args,
            es_step=es_step,
            eval_sampling_kwargs=eval_sampling_kwargs,
        )
        prompts, answers = task.get_batch()
        results = ray.get(
            [
                engines[i].generate_and_score.remote(
                    _engine_prompts(prompts, engine_paths[i]),
                    sampling_kwargs,
                    _engine_lora_specs(engine_pop_indices[i], engine_paths[i], es_step, len(prompts)),
                    task,
                    answers,
                    args,
                )
                for i in range(args.num_engines)
            ]
        )
        fitnesses = []
        info_by_key: dict[str, list[float]] = {}
        logs: list[str] = []
        for eng_fitness, info, eng_logs in results:
            fitnesses.append(np.asarray(eng_fitness, dtype=np.float64).reshape(loras_per_engine, len(prompts)))
            for key, value in info.items():
                info_by_key.setdefault(key, []).append(float(value))
            logs.extend(eng_logs)

        fitnesses_shaped = np.concatenate(fitnesses, axis=0)
        summary = summarize_fitness(fitnesses_shaped, normalize_with_std=args.normalize_with_std)
        last_summary = summary
        normalized = summary.normalized.astype(float).tolist()
        ray.get(
            engines[0].collective_rpc.remote(
                "apply_lora_es_update",
                args=(normalized, template.base_shapes, es_step, args),
            )
        )
        if args.num_engines > 1:
            broadcast_weights(ray, engines)

        elapsed = time.time() - start_time
        iter_elapsed = time.time() - iter_start
        info = {key: float(np.mean(values)) for key, values in info_by_key.items()}
        info.update(eval_info)
        print(
            "ITER:"
            f" iter = {es_step}"
            f" elapsed = {elapsed:.2f}s"
            f" iter_dt = {iter_elapsed:.3f}s"
            f" y_best = {summary.max:.4f}"
            f" y_mean = {summary.mean:.4f}"
            f" y_min = {summary.min:.4f}"
            f" std_norm = {summary.normalized_std:.4f}"
            f" prop_truncated = {info.get('prop_truncated', 0.0):.3f}"
            f" mean_tokens = {info.get('mean_token_length', 0.0):.1f}"
        )
        for text in logs[:2]:
            print(text)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "es_step": es_step,
                    "fitness/mean": summary.mean,
                    "fitness/min": summary.min,
                    "fitness/max": summary.max,
                    "fitness/normalized_std": summary.normalized_std,
                    **info,
                }
            )
        maybe_save_checkpoint(ray, engines, args=args, task=task, es_step=es_step)

    if last_summary is None:
        raise RuntimeError("EggRoll loop ended before producing a fitness summary.")
    return {
        "iterations": args.num_iterations,
        "best": last_summary.max,
        "mean": last_summary.mean,
    }


def run_eval(
    ray: Any,
    *,
    engines: list[Any],
    eval_task: MathTask | None,
    args: EggrollArgs,
    es_step: int,
    eval_sampling_kwargs: dict[str, Any],
) -> dict[str, float]:
    if eval_task is None or args.steps_per_eval <= 0 or es_step % args.steps_per_eval != 0:
        return {}

    prompts, answers = eval_task.get_eval_batch()
    requests_per_engine = int(np.ceil(len(prompts) / args.num_engines))
    refs = []
    for i, engine in enumerate(engines):
        start = i * requests_per_engine
        stop = min(start + requests_per_engine, len(prompts))
        if start >= stop:
            continue
        refs.append(
            engine.generate_and_score.remote(
                prompts[start:stop],
                eval_sampling_kwargs,
                None,
                eval_task,
                answers[start:stop],
                args,
            )
        )
    results = ray.get(refs)
    all_fitnesses = np.concatenate([np.asarray(fitness, dtype=np.float64) for fitness, _, _ in results])
    return {
        "eval/mean": float(np.mean(all_fitnesses)),
        "eval/max": float(np.max(all_fitnesses)),
    }


def launch_engines(ray: Any, *, cfg: LLMConfig, args: EggrollArgs) -> tuple[list[Any], list[Any]]:
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from llm.vllm import EggrollVLLMActor

    placement_groups = []
    strategies = []
    for _ in range(args.num_engines):
        bundles = [{"GPU": 1, "CPU": 2} for _ in range(args.tensor_parallel_size)]
        pg = placement_group(bundles, lifetime="detached", strategy="STRICT_PACK")
        ray.get(pg.ready())
        placement_groups.append(pg)
        if args.tensor_parallel_size == 1:
            strategies.append(
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
            )
        else:
            strategies.append(
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                )
            )

    loras_per_engine = args.population_size // args.num_engines
    enforce_eager = args.tensor_parallel_size > 1
    actors = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(EggrollVLLMActor).remote(
            model_name=cfg.policy.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_loras=loras_per_engine,
            lora_rank=args.lora_r,
            max_tokens=args.max_tokens,
            prompt_batch_size=args.prompt_batch_size,
            enforce_eager=enforce_eager,
        )
        for strategy in strategies
    ]
    return actors, placement_groups


def shutdown_engines(ray: Any, *, engines: list[Any], placement_groups: list[Any]) -> None:
    try:
        ray.get([engine.shutdown.remote() for engine in engines], timeout=60)
    except Exception:
        pass
    try:
        terminate_refs = [engine.__ray_terminate__.remote() for engine in engines]
        ray.wait(terminate_refs, timeout=30, num_returns=len(terminate_refs))
    except Exception:
        for engine in engines:
            try:
                ray.kill(engine, no_restart=True)
            except Exception:
                pass
    for pg in placement_groups:
        try:
            ray.util.remove_placement_group(pg)
        except Exception:
            pass


def init_worker_groups(ray: Any, engines: list[Any], *, args: EggrollArgs) -> None:
    master_info = ray.get(engines[0].collective_rpc.remote("get_transport_info", args=()))[0]
    master_address, master_port = master_info
    results = ray.get(
        [
            engines[i].collective_rpc.remote(
                "init_inter_engine_group",
                args=(master_address, master_port, i, args.num_engines),
            )
            for i in range(args.num_engines)
        ]
    )
    if not all(result[0] for result in results):
        raise RuntimeError("Failed to initialize at least one vLLM worker group.")


def setup_lora_generation(ray: Any, engines: list[Any], *, template: LoraTemplate) -> None:
    state_ref = ray.put(template.state_dict)
    shapes_ref = ray.put(template.base_shapes)
    config_ref = ray.put(template.config)
    ray.get([engines[i].setup_local_lora_generation.remote(state_ref, shapes_ref, config_ref, i) for i in range(len(engines))])


def broadcast_weights(ray: Any, engines: list[Any]) -> None:
    results = ray.get([engine.collective_rpc.remote("broadcast_all_weights", args=(0,)) for engine in engines])
    if all(result[0] for result in results):
        return
    state = ray.get(engines[0].collective_rpc.remote("get_model_state_dict", args=()))[0]
    state_ref = ray.put(state)
    ray.get([engine.collective_rpc.remote("set_model_state_dict", args=(state_ref,)) for engine in engines[1:]])


def maybe_save_checkpoint(ray: Any, engines: list[Any], *, args: EggrollArgs, task: Any, es_step: int) -> None:
    if args.save_freq is None:
        return
    if int(args.save_freq) == -1:
        should_save = es_step == args.num_iterations - 1
    else:
        should_save = int(args.save_freq) > 0 and (es_step + 1) % int(args.save_freq) == 0
    if not should_save:
        return

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Checkpointing requires safetensors.") from exc

    checkpoint_dir = Path(args.checkpoint_dir or "checkpoints") / f"checkpoint_step_{es_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = ray.get(engines[0].collective_rpc.remote("get_model_state_dict", args=()))[0]
    save_file(state, str(checkpoint_dir / "model_weights.safetensors"))
    task_state = task.state_dict() if hasattr(task, "state_dict") else {}
    with open(checkpoint_dir / "training_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "es_step": int(es_step),
                "args": dataclasses.asdict(args),
                "task_state": task_state,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _engine_prompts(prompts: list[str], adapter_paths: list[str]) -> list[str]:
    out: list[str] = []
    for _ in adapter_paths:
        out.extend(prompts)
    return out


def _engine_lora_specs(pop_indices: list[int], adapter_paths: list[str], es_step: int, num_prompts: int) -> list[tuple[str, int, str]]:
    specs: list[tuple[str, int, str]] = []
    for pop_idx, path in zip(pop_indices, adapter_paths, strict=True):
        spec = (f"adapter_{pop_idx}", int(pop_idx) + 1 + int(es_step) * 10000, path)
        specs.extend([spec] * int(num_prompts))
    return specs
