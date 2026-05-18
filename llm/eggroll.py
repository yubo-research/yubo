from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
from typing import Any

from llm.config import LLMConfig
from llm.console_observer import UnifiedConsoleManager
from llm.eggroll_engine import (
    EggrollArgs,
    _engine_lora_specs,
    init_worker_groups,
    launch_engines,
    setup_lora_generation,
    shutdown_engines,
    train_loop,
)
from llm.eggroll_support import (
    adapter_root_for as _adapter_root_for,
)
from llm.eggroll_support import (
    base_seed as _base_seed,
)
from llm.eggroll_support import (
    write_run_config as _write_run_config,
)
from llm.engine_pool import ensure_ray as _init_ray
from llm.engine_pool import sampling_kwargs as _sampling_kwargs
from llm.es import (
    num_iterations_from_budget,
    validate_eggroll_population,
)
from llm.registry import policy_uses_chat_template
from llm.tasks import MathTask, build_task


def run_eggroll(cfg: LLMConfig) -> dict[str, Any]:
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    missing = [module for module in ("ray", "transformers") if importlib.util.find_spec(module) is None]
    if missing:
        from llm.eggroll_support import eggroll_missing_runtime_message

        raise RuntimeError(eggroll_missing_runtime_message(missing))
    try:
        import ray
        from transformers import AutoTokenizer
    except ImportError as exc:
        from llm.eggroll_support import eggroll_missing_runtime_message

        raise RuntimeError(eggroll_missing_runtime_message(["ray", "transformers"])) from exc

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
        pretrain_lora_only=bool(cfg.pretrain_lora_only),
        pretrain_search_dim=int(cfg.pretrain_search_dim),
        vllm_enforce_eager=bool(cfg.vllm_enforce_eager),
        vllm_max_model_len=cfg.vllm_max_model_len,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_num_seqs=cfg.vllm_max_num_seqs,
        vllm_max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
        vllm_speculative_method=cfg.vllm_speculative_method,
        vllm_speculative_model=cfg.vllm_speculative_model,
        vllm_num_speculative_tokens=cfg.vllm_num_speculative_tokens,
    )
    validate_eggroll_population(
        population_size=args.population_size,
        num_engines=args.num_engines,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        pass_at_k=args.pass_at_k,
    )

    print(f"LLM_EGGROLL: model={cfg.policy.model_name} task={cfg.env.task_name} engines={args.num_engines} tp={args.tensor_parallel_size}")

    if args.pretrain_lora_only:
        from llm.lora import build_peft_lora_template

        template = build_peft_lora_template(
            model_name=cfg.policy.model_name,
            rank=args.lora_r,
            alpha=args.lora_alpha,
        )
    else:
        template = None  # Will build after launching engines

    tokenizer = AutoTokenizer.from_pretrained(cfg.policy.model_name, trust_remote_code=True)
    console = UnifiedConsoleManager()
    task = build_task(
        cfg.env,
        batch_size=args.prompt_batch_size,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        dataset_size=cfg.sub_dataset_size,
        tokenizer=tokenizer,
        apply_chat_template=policy_uses_chat_template(cfg.policy),
        console=console,
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

    engines, placement_groups = launch_engines(ray, cfg=cfg, args=args)
    try:
        init_worker_groups(ray, engines, args=args)

        if args.pretrain_lora_only:
            setup_lora_generation(ray, engines, template=template)
        else:
            # Architecture-agnostic discovery for universal subspace
            from llm.lora import build_universal_subspace_template

            params_meta = ray.get(engines[0].get_parameter_metadata.remote())[0]
            template = build_universal_subspace_template(
                parameters=params_meta,
                search_dim=args.pretrain_search_dim,
                seed=args.base_seed,
                lora_only=False,
            )
            print(f"UNIVERSAL: discovered {len(params_meta)} parameters, search_dim={args.pretrain_search_dim}")

        wandb_run = _maybe_init_wandb(args=args, cfg=cfg)
        result = train_loop(
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
        shutdown_engines(ray, engines=engines, placement_groups=placement_groups)


def _maybe_init_wandb(*, args: EggrollArgs, cfg: LLMConfig) -> Any | None:
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("use_wandb=true requires wandb.") from exc
    name = args.wandb_name or f"{cfg.env.task_name.replace(':', '_')}-{cfg.policy.policy_tag}-eggroll"
    return wandb.init(project=args.wandb_project, name=name, config=dataclasses.asdict(cfg))


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
        apply_chat_template=policy_uses_chat_template(cfg.policy),
    )


__all__ = ["EggrollArgs", "run_eggroll", "_engine_lora_specs"]
