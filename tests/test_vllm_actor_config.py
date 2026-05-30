import os

import pytest

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
    expected_gpu_mem = 0.55 if os.uname().sysname == "Darwin" else 0.85
    assert kwargs["gpu_memory_utilization"] == expected_gpu_mem
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


def test_vllm_actor_kwargs_accept_speculative_config():
    kwargs = get_llm_kwargs(
        VLLMActorConfig(
            model_name="google/gemma-4-E2B-it",
            tensor_parallel_size=1,
            max_loras=1,
            lora_rank=4,
            max_tokens=128,
            prompt_batch_size=1,
            enforce_eager=False,
            vllm_speculative_model="google/gemma-4-E2B-it-assistant",
            vllm_num_speculative_tokens=2,
        )
    )

    assert kwargs["speculative_config"] == {
        "model": "google/gemma-4-E2B-it-assistant",
        "num_speculative_tokens": 2,
    }
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "audio": 0, "video": 0}


def test_gemma4_lora_targets_discover_text_linear_children():
    torch = pytest.importorskip("torch")
    from llm.lora import _lora_target_modules

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
    _add_fake_gemma4_text_lora_layer(model.language_model.layers[0], torch)

    model.audio_tower = torch.nn.Module()
    model.audio_tower.model = torch.nn.Module()
    model.audio_tower.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    _add_fake_gemma4_wrapped_lora_layer(model.audio_tower.model.layers[0], torch)

    model.vision_tower = torch.nn.Module()
    model.vision_tower.encoder = torch.nn.Module()
    model.vision_tower.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    _add_fake_gemma4_wrapped_lora_layer(model.vision_tower.encoder.layers[0], torch)

    gemma4_targets = _lora_target_modules("google/gemma-4-E2B-it", base_model=model)

    assert "language_model.layers.0.self_attn.q_proj" in gemma4_targets
    assert "language_model.layers.0.self_attn.o_proj" in gemma4_targets
    assert "language_model.layers.0.mlp.gate_proj" in gemma4_targets
    assert "language_model.layers.0.mlp.down_proj" in gemma4_targets
    assert all("audio_tower" not in target for target in gemma4_targets)
    assert all("vision_tower" not in target for target in gemma4_targets)
    assert "q_proj" in _lora_target_modules("Qwen/Qwen3-1.7B")


def test_gemma4_vllm_update_target_allows_wrapped_linear_names():
    from llm.lora import vllm_dense_update_target

    qkv_param = object()
    peft_name = "base_model.model.model.layers.0.self_attn.q_proj.linear.base_layer.weight"
    target_param, rows = vllm_dense_update_target(
        peft_name=peft_name,
        weight_shape=(768, 768),
        peft_shapes_dict={peft_name: (768, 768)},
        vllm_params={"model.layers.0.self_attn.qkv_proj.base_layer.weight": qkv_param},
    )

    assert target_param is qkv_param
    assert rows == slice(0, 768)


def test_gemma4_vllm_update_target_normalizes_language_model_path():
    from llm.lora import vllm_dense_update_target

    qkv_param = object()
    for peft_name in (
        "base_model.model.model.layers.0.self_attn.q_proj.linear.base_layer.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.linear.base_layer.weight",
        "base_model.model.model.language_model.model.layers.0.self_attn.q_proj.linear.base_layer.weight",
    ):
        target_param, rows = vllm_dense_update_target(
            peft_name=peft_name,
            weight_shape=(768, 768),
            peft_shapes_dict={peft_name: (768, 768)},
            vllm_params={"language_model.model.layers.0.self_attn.qkv_proj.base_layer.weight": qkv_param},
        )

        assert target_param is qkv_param
        assert rows == slice(0, 768)


def _add_fake_gemma4_text_lora_layer(layer, torch):
    layer.self_attn = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(layer.self_attn, name, torch.nn.Linear(2, 2, bias=False))

    layer.mlp = torch.nn.Module()
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(layer.mlp, name, torch.nn.Linear(2, 2, bias=False))


def _add_fake_gemma4_wrapped_lora_layer(layer, torch):
    layer.self_attn = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        projection = torch.nn.Module()
        projection.linear = torch.nn.Linear(2, 2, bias=False)
        setattr(layer.self_attn, name, projection)

    layer.mlp = torch.nn.Module()
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = torch.nn.Module()
        projection.linear = torch.nn.Linear(2, 2, bias=False)
        setattr(layer.mlp, name, projection)
