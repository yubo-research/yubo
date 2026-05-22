from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from ops.uhd_config import UHDConfig
from problems.nanochat_dataloader import BinDataLoader
from problems.nanochat_lora import _NanochatSubspaceCodec
from problems.nanochat_obj import NanochatUHDVectorObjective, is_nanochat_env
from third_party.nanochat.gpt import GPT, GPTConfig


def test_nanochat_env_tag():
    assert is_nanochat_env("nanochat:synthetic") is True
    assert is_nanochat_env("nanochat:tinystories") is True
    assert is_nanochat_env("llm:math:gsm8k") is False


def test_gpt_initialization():
    # n_kv_head must be <= n_head and n_head must be divisible by n_kv_head
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
    model = GPT(config)
    model.init_weights()
    assert isinstance(model, torch.nn.Module)

    # Check weight initialization (should not be all zeros)
    # Target wte which is definitely non-zero after normal_ init
    assert torch.sum(torch.abs(model.transformer.wte.weight)) > 0


def test_nanochat_subspace_codec():
    # n_kv_head must be <= n_head and n_head must be divisible by n_kv_head
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)
    model = GPT(config)
    model.init_weights()

    # Use a larger dim to ensure we hit most parameters, or check total sum
    dim = 100
    codec = _NanochatSubspaceCodec(model, dim=dim, delta_scale=1.0, seed=42)

    assert codec.dim == dim
    assert len(codec.x0) == dim

    # Test apply/revert across the whole model
    x = np.random.randn(dim).astype(np.float64)

    # Sum of all weights before
    sum_before = sum(p.sum().item() for p in model.parameters())

    codec.apply(x)
    sum_after = sum(p.sum().item() for p in model.parameters())
    assert not np.isclose(sum_before, sum_after)

    codec.revert(x)
    sum_final = sum(p.sum().item() for p in model.parameters())
    assert np.isclose(sum_before, sum_final, atol=1e-5)


def test_bin_dataloader():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # Create dummy data: 100 tokens as uint16
        tokens = np.arange(100, dtype=np.uint16)
        tokens.tofile(f.name)
        temp_path = f.name

    try:
        b, t = 4, 8
        loader = BinDataLoader(temp_path, b=b, t=t)
        assert loader.num_tokens == 100

        x, y = loader.get_batch(seed=42, device="cpu")
        assert x.shape == (b, t)
        assert y.shape == (b, t)

        # Verify autoregressive target (y should be x shifted by 1)
        for i in range(b):
            val_x = x[i, 0].item()
            assert val_x < 100
            assert y[i, 0].item() == (val_x + 1)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_nanochat_objective_synthetic():
    # UHDConfig has many required fields
    # Using field names from ops/uhd_config.py
    cfg = UHDConfig(
        env_tag="nanochat:synthetic",
        policy_tag="nanochat:d12",
        num_rounds=1,
        problem_seed=42,
        noise_seed_0=0,
        lr=1e-4,
        num_dim_target=None,
        num_module_target=None,
        log_interval=1,
        accuracy_interval=1,
        target_accuracy=None,
        optimizer="mezo",
        batch_size=1,
        early_reject=None,
        be=None,
        enn=None,
        bszo_k=1,
        bszo_epsilon=1e-4,
        bszo_sigma_p_sq=1.0,
        bszo_sigma_e_sq=1.0,
        bszo_alpha=0.1,
        text_search_dim=16,
        num_envs=2,
        max_tokens=32,
    )

    obj = NanochatUHDVectorObjective(cfg)
    assert obj.dim == 16
    assert obj.num_envs == 2

    x = np.zeros(16)
    y, se = obj.evaluate(x, seed=123)

    assert isinstance(y, float)
    assert y < 0  # BPB is positive, so objective should be negative
    assert se == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
