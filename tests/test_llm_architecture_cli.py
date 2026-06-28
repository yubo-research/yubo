from __future__ import annotations

import json

from click.testing import CliRunner


def _dense_model(torch):
    nn = torch.nn
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    block = model.layers[0]
    block.self_attn = nn.Module()
    block.self_attn.q_proj = nn.Linear(2, 2, bias=False)
    block.self_attn.k_proj = nn.Linear(2, 2, bias=False)
    block.self_attn.v_proj = nn.Linear(2, 2, bias=False)
    block.self_attn.o_proj = nn.Linear(2, 2, bias=False)
    block.mlp = nn.Module()
    block.mlp.down_proj = nn.Linear(4, 2, bias=False)
    return model


def _router_model(torch):
    nn = torch.nn
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.gate = nn.Linear(2, 2, bias=False)
    return model


def test_llm_architecture_cli_inspect_outputs_json(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    from ops import llm_architecture

    monkeypatch.setattr(llm_architecture, "_load_empty_causal_lm", lambda *_args, **_kwargs: _dense_model(torch))

    result = CliRunner().invoke(
        llm_architecture.cli,
        ["inspect", "qwen3-1p7b-lora-r1", "--roles", "attention_q", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["policy"] == "qwen3-1p7b-lora-r1"
    assert payload["model_name"] == "Qwen/Qwen3-1.7B"
    assert payload["program"]["roles"] == ["attention_q"]
    assert payload["selected_modules"] == ["layers.0.self_attn.q_proj"]
    assert payload["direct_vllm_dense_update"]["supported"] is True


def test_llm_architecture_cli_reports_router_unsupported_for_direct_update(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    from ops import llm_architecture

    monkeypatch.setattr(llm_architecture, "_load_empty_causal_lm", lambda *_args, **_kwargs: _router_model(torch))

    result = CliRunner().invoke(
        llm_architecture.cli,
        ["inspect", "qwen3-1p7b-lora-r1", "--roles", "moe_router", "--expert-policy", "router"],
    )

    assert result.exit_code == 0, result.output
    assert "direct_vllm_dense_update: unsupported" in result.output
    assert "unsupported: layers.0.mlp.gate" in result.output


def test_llm_architecture_cli_can_skip_direct_update_support_check(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    from ops import llm_architecture

    monkeypatch.setattr(llm_architecture, "_load_empty_causal_lm", lambda *_args, **_kwargs: _router_model(torch))

    result = CliRunner().invoke(
        llm_architecture.cli,
        [
            "inspect",
            "qwen3-1p7b-lora-r1",
            "--roles",
            "moe_router",
            "--expert-policy",
            "router",
            "--no-direct-vllm-dense-update",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "direct_vllm_dense_update: not_checked" in result.output
    assert "unsupported: layers.0.mlp.gate" not in result.output


def test_llm_architecture_cli_inspect_config_reads_llm_update_fields(tmp_path, monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    from ops import llm_architecture

    monkeypatch.setattr(llm_architecture, "_load_empty_causal_lm", lambda *_args, **_kwargs: _dense_model(torch))
    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
llm_update_roles = ["attention_q"]
llm_update_max_targets = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(llm_architecture.cli, ["inspect-config", str(config), "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["program"]["roles"] == ["attention_q"]
    assert payload["program"]["max_targets"] == 1
    assert payload["selected_modules"] == ["layers.0.self_attn.q_proj"]
