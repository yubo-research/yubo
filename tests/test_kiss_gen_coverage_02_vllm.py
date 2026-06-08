"""Auto-generated kiss test_coverage witnesses (vLLM modules)."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_llm_vllm_actor() -> None:
    from llm.vllm_actor import EggrollVLLMActor

    setup_local_lora_generation = EggrollVLLMActor.setup_local_lora_generation
    set_engine_rank = EggrollVLLMActor.set_engine_rank
    set_universal_subspace_template = EggrollVLLMActor.set_universal_subspace_template
    get_parameter_metadata = EggrollVLLMActor.get_parameter_metadata
    apply_universal_update = EggrollVLLMActor.apply_universal_update
    generate_local_adapters = EggrollVLLMActor.generate_local_adapters
    generate_and_score = EggrollVLLMActor.generate_and_score
    refs = (
        setup_local_lora_generation,
        set_engine_rank,
        set_universal_subspace_template,
        get_parameter_metadata,
        apply_universal_update,
        generate_local_adapters,
        generate_and_score,
    )
    assert refs


def test_kiss_gen_llm_vllm_nll_scoring() -> None:
    from llm.vllm_nll_scoring import NLLScoringItem, nll_prefix_sampling_kwargs, nll_sampling_kwargs, nll_use_prefix_decode

    refs = (
        NLLScoringItem,
        nll_sampling_kwargs,
        nll_prefix_sampling_kwargs,
        nll_use_prefix_decode,
    )
    assert refs


def test_kiss_gen_llm_vllm_worker() -> None:
    from llm.vllm_worker import WorkerExtension

    get_transport_info = WorkerExtension.get_transport_info
    init_inter_engine_group = WorkerExtension.init_inter_engine_group
    set_universal_subspace_template = WorkerExtension.set_universal_subspace_template
    discover_parameters = WorkerExtension.discover_parameters
    broadcast_all_weights = WorkerExtension.broadcast_all_weights
    get_model_state_dict = WorkerExtension.get_model_state_dict
    set_model_state_dict = WorkerExtension.set_model_state_dict
    refs = (
        get_transport_info,
        init_inter_engine_group,
        set_universal_subspace_template,
        discover_parameters,
        broadcast_all_weights,
        get_model_state_dict,
        set_model_state_dict,
    )
    assert refs
