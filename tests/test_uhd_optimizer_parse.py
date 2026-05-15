"""Tests for UHD optimizer name parsing."""

from __future__ import annotations

import pytest

from ops.exp_uhd_parse import _parse_cfg


def test_parse_cfg_rejects_unsupported_uhd_optimizer():
    with pytest.raises(ValueError, match="Unsupported UHD optimizer 'eggroll'"):
        _parse_cfg(
            {
                "env_tag": "llm:math:gsm8k",
                "policy_tag": "qwen3-1p7b-lora-r1",
                "optimizer": "eggroll",
                "num_rounds": 1,
            }
        )
