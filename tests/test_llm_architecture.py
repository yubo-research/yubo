from __future__ import annotations

import pytest


def _dense_block(nn):
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = nn.Linear(4, 4, bias=False)
    block.self_attn.k_proj = nn.Linear(4, 4, bias=False)
    block.self_attn.v_proj = nn.Linear(4, 4, bias=False)
    block.self_attn.o_proj = nn.Linear(4, 4, bias=False)
    block.mlp = nn.Module()
    block.mlp.gate_proj = nn.Linear(4, 8, bias=False)
    block.mlp.up_proj = nn.Linear(4, 8, bias=False)
    block.mlp.down_proj = nn.Linear(8, 4, bias=False)
    return block


def _moe_expert(nn):
    expert = nn.Module()
    expert.gate_proj = nn.Linear(4, 8, bias=False)
    expert.up_proj = nn.Linear(4, 8, bias=False)
    expert.down_proj = nn.Linear(8, 4, bias=False)
    return expert


def _moe_block(nn):
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.gate = nn.Linear(4, 2, bias=False)
    block.mlp.experts = nn.ModuleList([_moe_expert(nn), _moe_expert(nn)])
    block.mlp.shared_expert = _moe_expert(nn)
    return block


def _hybrid_block(nn):
    block = nn.Module()
    block.mixer = nn.Module()
    block.mixer.in_proj = nn.Linear(4, 8, bias=False)
    block.mixer.dt_proj = nn.Linear(4, 4, bias=False)
    block.mixer.conv1d = nn.Conv1d(4, 4, kernel_size=3, bias=False)
    block.mixer.out_proj = nn.Linear(8, 4, bias=False)
    return block


def _gpt_neox_block(nn):
    block = nn.Module()
    block.attention = nn.Module()
    block.attention.query_key_value = nn.Linear(4, 12, bias=False)
    block.attention.dense = nn.Linear(4, 4, bias=False)
    block.mlp = nn.Module()
    block.mlp.dense_h_to_4h = nn.Linear(4, 16, bias=False)
    block.mlp.dense_4h_to_h = nn.Linear(16, 4, bias=False)
    return block


def test_llm_architecture_discovers_dense_transformer_roles() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Module()
    model.layers = nn.ModuleList([_dense_block(nn) for _ in range(3)])
    model.lm_head = nn.Linear(4, 10, bias=False)

    from llm.architecture import LLMUpdateProgram, discover_architecture_profile, lora_target_module_names, resolve_update_program

    profile = discover_architecture_profile(model)
    counts = profile.role_counts()

    assert counts["attention_q"] == 3
    assert counts["attention_o"] == 3
    assert counts["mlp_down"] == 3
    assert counts["lm_head"] == 1
    assert "layers.0.self_attn.q_proj" in lora_target_module_names(profile)

    middle_program = LLMUpdateProgram(roles=("attention_q",), layer_band="middle")
    targets = resolve_update_program(profile, middle_program)

    assert [target.name for target in targets] == ["layers.1.self_attn.q_proj.weight"]


def test_llm_architecture_discovers_gpt_neox_fused_attention_roles() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Module()
    model.gpt_neox = nn.Module()
    model.gpt_neox.layers = nn.ModuleList([_gpt_neox_block(nn) for _ in range(2)])

    from llm.architecture import LLMUpdateProgram, discover_architecture_profile, resolve_update_program

    profile = discover_architecture_profile(model)
    counts = profile.role_counts()

    assert counts["attention_qkv"] == 2
    assert counts["attention_o"] == 2
    assert counts["mlp_up"] == 2
    assert counts["mlp_down"] == 2

    targets = resolve_update_program(profile, LLMUpdateProgram(roles=("attention_qkv", "mlp_down"), layer_band="all"))

    assert [target.name for target in targets] == [
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
        "gpt_neox.layers.1.attention.query_key_value.weight",
        "gpt_neox.layers.1.mlp.dense_4h_to_h.weight",
    ]


def test_llm_architecture_separates_moe_router_routed_and_shared_experts() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Module()
    model.layers = nn.ModuleList([_moe_block(nn)])

    from llm.architecture import LLMUpdateProgram, discover_architecture_profile, resolve_update_program

    profile = discover_architecture_profile(model)

    assert profile.role_counts()["moe_router"] == 1
    assert profile.role_counts()["moe_expert_gate"] == 2
    assert profile.role_counts()["moe_shared_down"] == 1

    routed = resolve_update_program(
        profile,
        LLMUpdateProgram(roles=("moe_expert_down",), expert_policy="routed"),
    )
    shared = resolve_update_program(
        profile,
        LLMUpdateProgram(roles=("moe_shared_down",), expert_policy="shared"),
    )
    router = resolve_update_program(
        profile,
        LLMUpdateProgram(roles=("moe_router",), expert_policy="router"),
    )

    assert [target.expert_index for target in routed] == [0, 1]
    assert [target.name for target in shared] == ["layers.0.mlp.shared_expert.down_proj.weight"]
    assert [target.name for target in router] == ["layers.0.mlp.gate.weight"]


def test_llm_architecture_discovers_ssm_mixer_roles() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = nn.Module()
    model.layers = nn.ModuleList([_hybrid_block(nn)])

    from llm.architecture import discover_architecture_profile

    profile = discover_architecture_profile(model)

    assert profile.role_counts()["ssm_in"] == 1
    assert profile.role_counts()["ssm_dt"] == 1
    assert profile.role_counts()["ssm_conv"] == 1
    assert profile.role_counts()["ssm_out"] == 1


def test_llm_architecture_discovers_rwkv_style_mixer_roles() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    block = nn.Module()
    block.time_mix = nn.Module()
    block.time_mix.key = nn.Linear(4, 4, bias=False)
    model = nn.Module()
    model.blocks = nn.ModuleList([block])

    from llm.architecture import discover_architecture_profile

    profile = discover_architecture_profile(model)

    assert profile.role_counts()["rnn_mixer"] == 1


def test_llm_update_program_features_are_stable_and_role_sensitive() -> None:
    from llm.architecture import (
        ArchitectureProfile,
        LLMUpdateProgram,
        SemanticTarget,
        update_program_features,
    )

    profile = ArchitectureProfile(
        model_class="Tiny",
        targets=(
            SemanticTarget(
                name="layers.0.self_attn.q_proj.weight",
                module_name="layers.0.self_attn.q_proj",
                parameter_name="weight",
                role="attention_q",
                shape=(4, 4),
                module_class="Linear",
                layer_index=0,
            ),
            SemanticTarget(
                name="layers.0.mlp.down_proj.weight",
                module_name="layers.0.mlp.down_proj",
                parameter_name="weight",
                role="mlp_down",
                shape=(4, 8),
                module_class="Linear",
                layer_index=0,
            ),
        ),
    )

    q_features = update_program_features(profile, LLMUpdateProgram(roles=("attention_q",), seed=11))
    down_features = update_program_features(profile, LLMUpdateProgram(roles=("mlp_down",), seed=11))

    assert q_features == update_program_features(profile, LLMUpdateProgram(roles=("attention_q",), seed=11))
    assert q_features != down_features
    assert len(q_features) == len(down_features)
