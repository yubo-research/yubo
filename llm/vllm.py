from __future__ import annotations

import gc
import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

from llm.lora import add_dense_update, get_rng_noise, materialize_lora_adapters, vllm_dense_update_target


class WorkerExtension:
    def get_transport_info(self) -> tuple[str, int]:
        return _get_ip(), _get_open_port()

    def init_inter_engine_group(self, master_address: str, master_port: int, gpu_rank: int, world_size: int) -> bool:
        self.device = self.model_runner.device
        self.gpu_rank = int(gpu_rank)
        self.world_size = int(world_size)
        self.inter_pg = None
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup
        except ImportError:
            return True

        if get_tensor_model_parallel_rank() == 0:
            pg = StatelessProcessGroup.create(
                host=master_address,
                port=int(master_port),
                rank=int(gpu_rank),
                world_size=int(world_size),
            )
            self.inter_pg = PyNcclCommunicator(pg, device=self.device)
        return True

    def apply_lora_es_update(self, normalized_fitnesses: list[float], peft_shapes_dict: dict[str, tuple[int, ...]], es_step: int, args: Any) -> bool:
        import torch

        if getattr(self, "gpu_rank", 0) != 0:
            return False

        normalized = [float(x) for x in normalized_fitnesses]
        vllm_params = dict(self.model_runner.model.named_parameters())
        pop_step = int(es_step) // int(args.steps_per_adapter)
        chunk_size = _update_chunk_size(int(args.population_size), int(args.lora_r))

        for layer_idx, (peft_name, weight_shape_raw) in enumerate(peft_shapes_dict.items()):
            weight_shape = tuple(int(dim) for dim in weight_shape_raw)
            lora_b_shape = (weight_shape[0], int(args.lora_r))
            lora_a_shape = (int(args.lora_r), weight_shape[1])
            layer_update = torch.zeros(weight_shape, device=self.device, dtype=torch.float32)

            for chunk_start in range(0, int(args.population_size) // 2, chunk_size):
                chunk_end = min(chunk_start + chunk_size, int(args.population_size) // 2)
                noise_a_list = []
                noise_b_list = []
                fitness_diffs = []

                for pop_pair_idx in range(chunk_start, chunk_end):
                    pop_idx_1 = pop_pair_idx * 2
                    pop_idx_2 = pop_pair_idx * 2 + 1
                    fitness_diffs.append(normalized[pop_idx_1] - normalized[pop_idx_2])
                    noise_a, noise_b = get_rng_noise(
                        base_seed=int(args.base_seed),
                        num_pop_pairs=int(args.population_size) // 2,
                        pop_pair_idx=pop_pair_idx,
                        num_layers=len(peft_shapes_dict),
                        layer_idx=layer_idx,
                        step=pop_step,
                        shapes=[lora_a_shape, lora_b_shape],
                    )
                    noise_a_list.append(noise_a)
                    noise_b_list.append(noise_b)

                noise_a_batch = torch.stack(noise_a_list).to(self.device) * math.sqrt(float(args.sigma))
                noise_b_batch = torch.stack(noise_b_list).to(self.device) * math.sqrt(float(args.sigma) / float(args.lora_r))
                fitness_diffs_tensor = torch.tensor(fitness_diffs, device=self.device, dtype=noise_a_batch.dtype)

                if int(args.lora_r) == 1:
                    noise_b_vec = noise_b_batch.squeeze(2)
                    noise_a_vec = noise_a_batch.squeeze(1)
                    weighted_b = noise_b_vec * fitness_diffs_tensor.unsqueeze(1)
                    weighted_noise = torch.mm(weighted_b.t(), noise_a_vec)
                else:
                    noise_batch = torch.bmm(noise_b_batch, noise_a_batch)
                    weighted_noise = (noise_batch * fitness_diffs_tensor.view(-1, 1, 1)).sum(dim=0)
                    del noise_batch

                layer_update.add_(weighted_noise)
                del noise_a_batch, noise_b_batch, weighted_noise

            gradient = layer_update * (float(args.learning_rate) / (int(args.population_size) * float(args.sigma) + 1e-8))
            if bool(args.scale_lr_in_grad):
                gradient *= math.sqrt(int(args.population_size))
            del layer_update

            target_param, slice_obj = vllm_dense_update_target(
                peft_name=peft_name,
                weight_shape=weight_shape,
                peft_shapes_dict=peft_shapes_dict,
                vllm_params=vllm_params,
            )
            add_dense_update(target_param, slice_obj, gradient)
            del gradient
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        return True

    def broadcast_all_weights(self, src_rank: int) -> bool:
        try:
            import torch
        except ImportError:
            return False

        if not getattr(self, "inter_pg", None):
            return False
        for _, param in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def get_model_state_dict(self) -> dict[str, Any]:
        return {name: param.detach().cpu().clone() for name, param in self.model_runner.model.named_parameters()}

    def set_model_state_dict(self, state_dict: dict[str, Any]) -> bool:
        import torch

        params = dict(self.model_runner.model.named_parameters())
        for name, value in state_dict.items():
            if name in params:
                params[name].data.copy_(value.to(params[name].device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True


class EggrollVLLMActor:
    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int,
        max_loras: int,
        lora_rank: int,
        max_tokens: int,
        prompt_batch_size: int,
        enforce_eager: bool,
    ) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_FUSED_MOE_CHUNK_SIZE", str(16 * 2048))
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from vllm import LLM

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=int(tensor_parallel_size),
            distributed_executor_backend="ray",
            worker_extension_cls="llm.vllm.WorkerExtension",
            dtype="auto",
            enable_prefix_caching=True,
            enforce_eager=bool(enforce_eager),
            enable_lora=True,
            max_loras=int(max_loras),
            max_lora_rank=max(int(lora_rank), 8),
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            max_num_seqs=512,
            max_model_len=max(1024, 512 + int(max_tokens)),
            max_num_batched_tokens=max(1, int(prompt_batch_size)) * 2048,
            enable_chunked_prefill=True,
            load_format="auto",
        )
        self.rank = 0
        self.adapter_root = Path(os.getenv("YUBO_LORA_POPULATION_PATH", "/dev/shm/yubo_llm_lora_population"))
        self.peft_state_dict = None
        self.peft_shapes_dict = None
        self.lora_config_dict = None

    def collective_rpc(self, method: str, args: tuple[Any, ...] = ()) -> Any:
        return self.llm.collective_rpc(method, args=args)

    def shutdown(self) -> bool:
        return bool(self._shutdown_vllm())

    def __ray_shutdown__(self) -> None:
        # Called on graceful actor termination (e.g. __ray_terminate__). Best-effort cleanup.
        self._shutdown_vllm()

    def _shutdown_vllm(self) -> bool:
        llm = getattr(self, "llm", None)
        if llm is None:
            return True
        try:
            shutdown_fn = getattr(llm, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()
                return True
            engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
            if engine is not None:
                for name in ("shutdown", "shutdown_background_loop"):
                    fn = getattr(engine, name, None)
                    if callable(fn):
                        fn()
            return True
        except Exception:
            return False
        finally:
            try:
                del self.llm
            except Exception:
                pass
            gc.collect()

    def setup_local_lora_generation(
        self,
        peft_state_dict: dict[str, Any],
        peft_shapes_dict: dict[str, tuple[int, ...]],
        lora_config_dict: dict[str, Any],
        rank: int,
    ) -> bool:
        self.rank = int(rank)
        self.peft_state_dict = peft_state_dict
        self.peft_shapes_dict = peft_shapes_dict
        self.lora_config_dict = lora_config_dict
        return True

    def generate_local_adapters(self, population_indices: list[int], es_step: int, args: Any) -> list[str]:
        if self.peft_state_dict is None or self.peft_shapes_dict is None or self.lora_config_dict is None:
            raise RuntimeError("LoRA generation was not initialized on this actor.")
        return materialize_lora_adapters(
            adapter_root=self.adapter_root,
            rank=self.rank,
            population_indices=population_indices,
            es_step=int(es_step),
            args=args,
            peft_state_dict=self.peft_state_dict,
            peft_shapes_dict=self.peft_shapes_dict,
            lora_config_dict=self.lora_config_dict,
        )

    def generate_and_score(
        self,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        task_obj: Any,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(**sampling_params_kwargs)
        lora_requests = None
        if lora_request_specs is not None:
            lora_requests = [
                LoRARequest(lora_name=name, lora_int_id=int(lora_id), lora_path=path)
                for name, lora_id, path in lora_request_specs
            ]

        request_outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_requests,
            use_tqdm=True,
        )
        return score_request_outputs(request_outputs, prompts=prompts, task_obj=task_obj, answers=answers, pass_at_k=bool(args.pass_at_k))


def score_request_outputs(
    request_outputs: list[Any],
    *,
    prompts: list[str],
    task_obj: Any,
    answers: list[Any],
    pass_at_k: bool,
) -> tuple[list[float], dict[str, float], list[str]]:
    fitness_list: list[float] = []
    distinct_counts: list[int] = []
    total_responses = 0
    num_truncated = 0
    mean_char_lengths: list[float] = []
    mean_token_lengths: list[float] = []
    responses_for_logging: list[str] = []
    all_sample_stds: list[float] = []
    all_pass_at_k_fitnesses: list[float] = []
    all_mean_fitnesses: list[float] = []
    num_prompts = len(answers)
    pop_responses_buffer = ""

    for i, output in enumerate(request_outputs):
        prompt_idx = i % num_prompts
        pop_idx = i // num_prompts
        gt_answer = answers[prompt_idx]
        responses = [sample.text for sample in output.outputs]
        truncateds = [sample.finish_reason == "length" for sample in output.outputs]
        fit, model_answers, sample_fitnesses = task_obj.score(responses, truncateds, gt_answer, pass_at_k=pass_at_k)
        sample_fitnesses = np.asarray(sample_fitnesses, dtype=np.float64)

        model_answers_set = set()
        for model_answer in model_answers:
            if model_answer is not None:
                model_answers_set.add(tuple(model_answer) if isinstance(model_answer, list) else model_answer)

        sample_char_lens = []
        sample_token_lens = []
        if pop_idx < 2 and prompt_idx < 3:
            current_prompt_log = f"\n[PROMPT {prompt_idx}]: {prompts[i]}\n"
        for j, sample in enumerate(output.outputs):
            text = sample.text
            if sample.finish_reason == "length":
                num_truncated += 1
            sample_char_lens.append(len(text))
            sample_token_lens.append(len(sample.token_ids))
            total_responses += 1
            if pop_idx < 2 and prompt_idx < 3:
                sample_fit = float(sample_fitnesses[j]) if j < len(sample_fitnesses) else float(fit)
                current_prompt_log += f"\n------SAMPLE {j + 1}: {text} || FIT={sample_fit}\n"

        if pop_idx < 2 and prompt_idx < 3:
            pop_responses_buffer += current_prompt_log
        if (i + 1) % num_prompts == 0 and pop_responses_buffer:
            responses_for_logging.append(f"-----POP {pop_idx} BATCH LOG-----\n{pop_responses_buffer}")
            pop_responses_buffer = ""

        fitness_list.append(float(fit))
        if len(sample_fitnesses) > 0:
            all_pass_at_k_fitnesses.append(float(np.max(sample_fitnesses)))
            all_mean_fitnesses.append(float(np.mean(sample_fitnesses)))
            if len(sample_fitnesses) > 1:
                all_sample_stds.append(float(np.std(sample_fitnesses)))
        else:
            all_pass_at_k_fitnesses.append(float(fit))
            all_mean_fitnesses.append(float(fit))
        distinct_counts.append(len(model_answers_set))
        mean_char_lengths.append(float(np.mean(sample_char_lens)))
        mean_token_lengths.append(float(np.mean(sample_token_lens)))

    info = {
        "total_responses": float(total_responses),
        "prop_truncated": float(num_truncated / total_responses) if total_responses else 0.0,
        "mean_char_length": float(np.mean(mean_char_lengths)) if mean_char_lengths else 0.0,
        "mean_token_length": float(np.mean(mean_token_lengths)) if mean_token_lengths else 0.0,
        "mean_distinct_counts": float(np.mean(distinct_counts)) if distinct_counts else 0.0,
        "std_in_samples": float(np.mean(all_sample_stds)) if all_sample_stds else 0.0,
        "pass_at_k_fitness": float(np.mean(all_pass_at_k_fitnesses)) if all_pass_at_k_fitnesses else 0.0,
        "mean_sample_fitness": float(np.mean(all_mean_fitnesses)) if all_mean_fitnesses else 0.0,
    }
    return fitness_list, info, responses_for_logging


class TextVLLMActor(EggrollVLLMActor):
    pass


def _get_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _update_chunk_size(population_size: int, lora_rank: int) -> int:
    if lora_rank <= 2:
        return min(128, population_size // 2)
    if lora_rank <= 8:
        return min(64, population_size // 2)
    return min(32, population_size // 2)


__all__ = [
    "EggrollVLLMActor",
    "TextVLLMActor",
    "WorkerExtension",
    "score_request_outputs",
]
