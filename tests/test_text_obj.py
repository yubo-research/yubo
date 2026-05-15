from __future__ import annotations

import numpy as np
import pytest


def test_text_uhd_config_file_loads():
    from pathlib import Path

    from ops.exp_uhd import _load_toml_config, _parse_cfg

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _parse_cfg(_load_toml_config(str(repo_root / "configs/uhd/text/gsm8k_qwen3_1p7b_mezo.toml")))

    assert cfg.env_tag == "llm:math:gsm8k"
    assert cfg.policy_tag == "qwen3-1p7b-lora-r1"
    assert cfg.optimizer == "mezo"
    assert cfg.text_search_dim == 128


def test_text_tags_route_to_uhd_vector_objective(monkeypatch):
    import problems.text_obj as text_obj
    from ops.exp_uhd import _parse_cfg
    from problems.uhd_obj import (
        build_uhd_vector_objective,
        supports_uhd_vector_objective,
    )

    class FakeTextObjective:
        dim = 5
        x0 = np.zeros(5, dtype=np.float64)
        steps_per_episode = 2
        num_envs = 1

        def __init__(self, cfg):
            self.cfg = cfg
            self.configured = 0

        def configure_embedding(self, num_probes):
            self.configured = int(num_probes)

    monkeypatch.setattr(text_obj, "TextObjective", FakeTextObjective)
    cfg = _parse_cfg(
        {
            "env_tag": "llm:zeros",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "num_rounds": 1,
            "text_search_dim": 5,
        }
    )

    assert supports_uhd_vector_objective("llm:zeros") is True
    built = build_uhd_vector_objective(cfg, embed_num_probes=3)

    assert built.source == "text"
    assert built.objective.dim == 5
    assert built.objective.configured == 3


def test_text_objective_requires_policy_tag():
    from ops.exp_uhd import _parse_cfg
    from problems.uhd_obj import build_uhd_vector_objective

    cfg = _parse_cfg({"env_tag": "llm:zeros", "num_rounds": 1})

    with pytest.raises(ValueError, match="UHD text objectives require policy_tag"):
        build_uhd_vector_objective(cfg)


def test_uhd_text_config_fields_parse():
    from ops.exp_uhd import _parse_cfg

    cfg = _parse_cfg(
        {
            "env_tag": "llm:zeros",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "num_rounds": 1,
            "max_tokens": 64,
            "temperature": 0.2,
            "samples_per_prompt": 2,
            "prompt_batch_size": 3,
            "pass_at_k": True,
            "num_gpus": 2,
            "num_engines": 1,
            "tensor_parallel_size": 1,
            "sub_dataset_size": 8,
            "hf_home": "/tmp/hf",
            "text_search_dim": 7,
            "text_delta_scale": 0.5,
            "text_basis_max_tensors": 4,
            "vllm_max_model_len": 4096,
            "vllm_gpu_memory_utilization": 0.82,
            "vllm_max_num_seqs": 8,
            "vllm_max_num_batched_tokens": 4096,
        }
    )

    assert cfg.max_tokens == 64
    assert cfg.temperature == 0.2
    assert cfg.samples_per_prompt == 2
    assert cfg.prompt_batch_size == 3
    assert cfg.pass_at_k is True
    assert cfg.num_gpus == 2
    assert cfg.num_engines == 1
    assert cfg.tensor_parallel_size == 1
    assert cfg.sub_dataset_size == 8
    assert cfg.hf_home == "/tmp/hf"
    assert cfg.text_search_dim == 7
    assert cfg.text_delta_scale == 0.5
    assert cfg.text_basis_max_tensors == 4
    assert cfg.vllm_max_model_len == 4096
    assert cfg.vllm_gpu_memory_utilization == 0.82
    assert cfg.vllm_max_num_seqs == 8
    assert cfg.vllm_max_num_batched_tokens == 4096


def test_lora_subspace_codec_searches_b_matrices_only():
    torch = pytest.importorskip("torch")

    from problems.text_obj import _LoraSubspaceCodec

    template = type(
        "Template",
        (),
        {
            "state_dict": {
                "layer.lora_A.default.weight": torch.ones((2, 3), dtype=torch.float32),
                "layer.lora_B.default.weight": torch.zeros((4, 2), dtype=torch.float32),
            },
            "config": {},
        },
    )()
    codec = _LoraSubspaceCodec(template, dim=4, delta_scale=1.0, seed=0, basis_max_tensors=None)

    decoded = codec.decode(np.ones(4, dtype=np.float64))

    assert codec.num_total_tensors == 1
    assert torch.all(decoded["layer.lora_A.default.weight"] == 1.0)
    assert torch.any(decoded["layer.lora_B.default.weight"] != 0.0)
