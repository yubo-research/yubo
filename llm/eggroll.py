from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm.config import LLMConfig
from llm.eggroll_support import adapter_root_for as _adapter_root_for
from llm.eggroll_support import base_seed as _base_seed
from llm.eggroll_support import eggroll_missing_runtime_message as _missing_runtime_message
from llm.eggroll_support import write_run_config as _write_run_config
from llm.engine_pool import ensure_ray as _init_ray
from llm.engine_pool import sampling_kwargs as _sampling_kwargs
from llm.es import num_iterations_from_budget, summarize_fitness, validate_eggroll_population
from llm.lora import LoraTemplate, build_peft_lora_template
from llm.tasks import MathTask, build_task


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


def run_eggroll(cfg: LLMConfig) -> dict[str, Any]:
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    missing = [module for module in ("ray", "transformers") if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(_missing_runtime_message(missing))
    try:
        import ray
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(_missing_runtime_message(["ray", "transformers"])) from exc

    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(exp_dir, cfg)
    os.environ.setdefault("YUBO_LORA_POPULATION_PATH", _adapter_root_for(exp_dir))

    _init_ray(ray)
    tensor_parallel_size = int(cfg.tensor_parallel_size or cfg.policy.tensor_parallel_size)
    total_gpus = int(cfg.num_gpus or ray.cluster_resources().get("GPU", 0))
    if total_gpus < 1:
        raise RuntimeError("Ray reports 0 GPUs; vLLM EggRoll requires at least one GPU.")
    num_engines = int(cfg.num_engines or (total_gpus // tensor_parallel_size))
    args = EggrollArgs(
        base_seed=_base_seed(cfg),
        population_size=int(cfg.population_size),
        num_iterations=num_iterations_from_budget(
            num_rounds=cfg.num_rounds,
            total_timesteps=cfg.total_timesteps,
            population_size=cfg.population_size,
            prompt_batch_size=cfg.prompt_batch_size,
        ),
        sigma=float(cfg.sigma),
        learning_rate=float(cfg.lr),
        lora_r=int(cfg.policy.lora_rank),
        lora_alpha=int(cfg.policy.lora_alpha),
        steps_per_adapter=int(cfg.steps_per_adapter),
        max_tokens=int(cfg.max_tokens),
        temperature=float(cfg.temperature),
        samples_per_prompt=int(cfg.samples_per_prompt),
        prompt_batch_size=int(cfg.prompt_batch_size),
        pass_at_k=bool(cfg.pass_at_k),
        normalize_with_std=bool(cfg.normalize_with_std),
        scale_lr_in_grad=bool(cfg.scale_lr_in_grad),
        num_gpus=total_gpus,
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        steps_per_eval=int(cfg.steps_per_eval),
        eval_batch_size=int(cfg.eval_batch_size),
        save_freq=cfg.save_freq,
        checkpoint_dir=cfg.checkpoint_dir or str(exp_dir / "checkpoints"),
        use_wandb=bool(cfg.use_wandb),
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    validate_eggroll_population(
        population_size=args.population_size,
        num_engines=args.num_engines,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        pass_at_k=args.pass_at_k,
    )

    print(f"LLM_EGGROLL: model={cfg.policy.model_name} task={cfg.env.task_name} engines={args.num_engines} tp={args.tensor_parallel_size}")
    template = build_peft_lora_template(
        model_name=cfg.policy.model_name,
        rank=args.lora_r,
        alpha=args.lora_alpha,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy.model_name, trust_remote_code=True)
    task = build_task(
        cfg.env,
        batch_size=args.prompt_batch_size,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        dataset_size=cfg.sub_dataset_size,
        tokenizer=tokenizer,
        apply_chat_template=False,
    )
    sampling_kwargs = _sampling_kwargs(
        tokenizer=tokenizer,
        temperature=args.temperature,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
    )
    eval_task = _build_eval_task(cfg, tokenizer=tokenizer, args=args)
    eval_sampling_kwargs = _sampling_kwargs(
        tokenizer=tokenizer,
        temperature=args.temperature,
        seed=args.base_seed + 12345,
        max_tokens=args.max_tokens,
        n=1,
    )

    engines, placement_groups = _launch_engines(ray, cfg=cfg, args=args)
    try:
        _init_worker_groups(ray, engines, args=args)
        _setup_lora_generation(ray, engines, template=template)
        wandb_run = _maybe_init_wandb(args=args, cfg=cfg)
        result = _train_loop(
            ray,
            engines=engines,
            args=args,
            template=template,
            task=task,
            eval_task=eval_task,
            sampling_kwargs=sampling_kwargs,
            eval_sampling_kwargs=eval_sampling_kwargs,
            wandb_run=wandb_run,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return result
    finally:
        _shutdown_engines(ray, engines=engines, placement_groups=placement_groups)


def _train_loop(
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

        eval_info = _run_eval(
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
        ray.get(engines[0].collective_rpc.remote("apply_lora_es_update", args=(normalized, template.base_shapes, es_step, args)))
        if args.num_engines > 1:
            _broadcast_weights(ray, engines)

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
        _maybe_save_checkpoint(ray, engines, args=args, task=task, es_step=es_step)

    if last_summary is None:
        raise RuntimeError("EggRoll loop ended before producing a fitness summary.")
    return {
        "iterations": args.num_iterations,
        "best": last_summary.max,
        "mean": last_summary.mean,
    }


def _run_eval(
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
    return {"eval/mean": float(np.mean(all_fitnesses)), "eval/max": float(np.max(all_fitnesses))}


def _launch_engines(ray: Any, *, cfg: LLMConfig, args: EggrollArgs) -> tuple[list[Any], list[Any]]:
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


def _shutdown_engines(ray: Any, *, engines: list[Any], placement_groups: list[Any]) -> None:
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


def _init_worker_groups(ray: Any, engines: list[Any], *, args: EggrollArgs) -> None:
    master_info = ray.get(engines[0].collective_rpc.remote("get_transport_info", args=()))[0]
    master_address, master_port = master_info
    results = ray.get(
        [engines[i].collective_rpc.remote("init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)) for i in range(args.num_engines)]
    )
    if not all(result[0] for result in results):
        raise RuntimeError("Failed to initialize at least one vLLM worker group.")


def _setup_lora_generation(ray: Any, engines: list[Any], *, template: LoraTemplate) -> None:
    state_ref = ray.put(template.state_dict)
    shapes_ref = ray.put(template.base_shapes)
    config_ref = ray.put(template.config)
    ray.get([engines[i].setup_local_lora_generation.remote(state_ref, shapes_ref, config_ref, i) for i in range(len(engines))])


def _broadcast_weights(ray: Any, engines: list[Any]) -> None:
    results = ray.get([engine.collective_rpc.remote("broadcast_all_weights", args=(0,)) for engine in engines])
    if all(result[0] for result in results):
        return
    state = ray.get(engines[0].collective_rpc.remote("get_model_state_dict", args=()))[0]
    state_ref = ray.put(state)
    ray.get([engine.collective_rpc.remote("set_model_state_dict", args=(state_ref,)) for engine in engines[1:]])


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


def _build_eval_task(cfg: LLMConfig, *, tokenizer: Any, args: EggrollArgs) -> MathTask | None:
    if cfg.env.task_kind != "math" or args.steps_per_eval <= 0:
        return None
    answer_format = "answer_tags" if cfg.env.answer_format == "answer_tags" else "none"
    return MathTask(
        batch_size=args.eval_batch_size,
        dataset_name="math-eval",
        seed=args.base_seed + 12345,
        answer_format=answer_format,
        tokenizer=tokenizer,
        apply_chat_template=False,
    )


def _maybe_init_wandb(*, args: EggrollArgs, cfg: LLMConfig) -> Any | None:
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("use_wandb=true requires wandb.") from exc
    name = args.wandb_name or f"{cfg.env.task_name.replace(':', '_')}-{cfg.policy.policy_tag}-eggroll"
    return wandb.init(project=args.wandb_project, name=name, config=dataclasses.asdict(cfg))


def _maybe_save_checkpoint(ray: Any, engines: list[Any], *, args: EggrollArgs, task: Any, es_step: int) -> None:
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


__all__ = ["EggrollArgs", "run_eggroll"]
