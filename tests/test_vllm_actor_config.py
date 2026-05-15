from llm.vllm_actor_config import VLLMActorConfig, get_llm_kwargs


def test_vllm_actor_kwargs_use_conservative_sequence_defaults():
    kwargs = get_llm_kwargs(
        VLLMActorConfig(
            model_name="test-model",
            tensor_parallel_size=1,
            max_loras=1,
            lora_rank=4,
            max_tokens=512,
            prompt_batch_size=1,
            samples_per_prompt=1,
            enforce_eager=False,
        )
    )

    assert kwargs["max_num_seqs"] == 8
    assert kwargs["gpu_memory_utilization"] == 0.85
    assert kwargs["max_num_batched_tokens"] == 2048


def test_vllm_actor_kwargs_accept_resource_overrides():
    kwargs = get_llm_kwargs(
        VLLMActorConfig(
            model_name="test-model",
            tensor_parallel_size=1,
            max_loras=1,
            lora_rank=4,
            max_tokens=512,
            prompt_batch_size=1,
            enforce_eager=False,
            vllm_max_model_len=4096,
            vllm_gpu_memory_utilization=0.82,
            vllm_max_num_seqs=4,
            vllm_max_num_batched_tokens=4096,
        )
    )

    assert kwargs["max_model_len"] == 4096
    assert kwargs["gpu_memory_utilization"] == 0.82
    assert kwargs["max_num_seqs"] == 4
    assert kwargs["max_num_batched_tokens"] == 4096
