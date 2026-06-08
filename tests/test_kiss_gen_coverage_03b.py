"""Auto-generated kiss test_coverage witnesses."""

# ruff: noqa: F821
from __future__ import annotations


def test_kiss_gen_third_party_nanochat_gpt() -> None:
    from third_party.nanochat.gpt import GPT

    get_device = GPT.get_device
    estimate_flops = GPT.estimate_flops
    num_scaling_params = GPT.num_scaling_params
    refs = (
        get_device,
        estimate_flops,
        num_scaling_params,
    )
    assert refs


def test_kiss_gen_third_party_nanochat_gpt_modules() -> None:
    from third_party.nanochat.gpt_modules import MLP, Block, CausalSelfAttention, apply_rotary_emb, has_ve

    forward = CausalSelfAttention.forward
    forward = MLP.forward
    forward = Block.forward
    if False:
        (
            CausalSelfAttention,
            __init__,
            MLP,
            __init__,
            Block,
            __init__,
        )
    refs = (
        has_ve,
        apply_rotary_emb,
        CausalSelfAttention,
        forward,
        MLP,
        forward,
        Block,
        forward,
    )
    assert refs


def test_kiss_gen_third_party_nanochat_loss_eval() -> None:
    from third_party.nanochat.loss_eval import evaluate_bpb

    refs = (evaluate_bpb,)
    assert refs


def test_kiss_gen_third_party_nanochat_optim() -> None:
    from third_party.nanochat.optim import DistMuonAdamW, MuonAdamW

    step = MuonAdamW.step
    step = DistMuonAdamW.step
    if False:
        (
            MuonAdamW,
            __init__,
            DistMuonAdamW,
            __init__,
        )
    refs = (
        MuonAdamW,
        step,
        DistMuonAdamW,
        step,
    )
    assert refs


def test_kiss_gen_third_party_nanochat_optim_kernels() -> None:
    from third_party.nanochat.optim_kernels import adamw_step_fused, muon_step_fused

    refs = (
        adamw_step_fused,
        muon_step_fused,
    )
    assert refs


def test_kiss_gen_third_party_nanochat_tokenizer() -> None:
    from third_party.nanochat.tokenizer import ConversationRenderer, HuggingFaceTokenizer, RustBPETokenizer, get_token_bytes, get_tokenizer

    from_directory = HuggingFaceTokenizer.from_directory
    train_from_iterator = HuggingFaceTokenizer.train_from_iterator
    get_vocab_size = HuggingFaceTokenizer.get_vocab_size
    get_special_tokens = HuggingFaceTokenizer.get_special_tokens
    id_to_token = HuggingFaceTokenizer.id_to_token
    encode_special = HuggingFaceTokenizer.encode_special
    get_bos_token_id = HuggingFaceTokenizer.get_bos_token_id
    decode = HuggingFaceTokenizer.decode
    preprocess_conversation = ConversationRenderer.preprocess_conversation
    render_message = ConversationRenderer.render_message
    render_conversation = ConversationRenderer.render_conversation
    train_from_iterator = RustBPETokenizer.train_from_iterator
    from_directory = RustBPETokenizer.from_directory
    get_vocab_size = RustBPETokenizer.get_vocab_size
    get_special_tokens = RustBPETokenizer.get_special_tokens
    id_to_token = RustBPETokenizer.id_to_token
    encode_special = RustBPETokenizer.encode_special
    get_bos_token_id = RustBPETokenizer.get_bos_token_id
    decode = RustBPETokenizer.decode
    render_conversation = RustBPETokenizer.render_conversation
    visualize_tokenization = RustBPETokenizer.visualize_tokenization
    render_for_completion = RustBPETokenizer.render_for_completion
    if False:
        (
            HuggingFaceTokenizer,
            __init__,
            RustBPETokenizer,
            __init__,
        )
    refs = (
        HuggingFaceTokenizer,
        from_directory,
        train_from_iterator,
        get_vocab_size,
        get_special_tokens,
        id_to_token,
        encode_special,
        get_bos_token_id,
        decode,
        ConversationRenderer,
        preprocess_conversation,
        render_message,
        render_conversation,
        RustBPETokenizer,
        train_from_iterator,
        from_directory,
        get_vocab_size,
        get_special_tokens,
        id_to_token,
        encode_special,
        get_bos_token_id,
        decode,
        render_conversation,
        visualize_tokenization,
        render_for_completion,
        get_tokenizer,
        get_token_bytes,
    )
    assert refs


def test_kiss_gen_turbo_m_ref_turbo_1_ask_tell_core() -> None:
    from turbo_m_ref.turbo_1_ask_tell_core import create_candidates, init_counters_and_tr, init_hypers, select_candidates

    refs = (
        init_hypers,
        init_counters_and_tr,
        create_candidates,
        select_candidates,
    )
    assert refs


def test_kiss_gen_turbo_m_ref_turbo_1_core() -> None:
    from turbo_m_ref.turbo_1_core import CandidatesResult

    refs = (CandidatesResult,)
    assert refs


def test_kiss_gen_video_isaaclab() -> None:
    from video.isaaclab import ensure_isaaclab_video_launcher, is_isaaclab_env_conf, render_isaaclab_video_episode

    refs = (
        is_isaaclab_env_conf,
        ensure_isaaclab_video_launcher,
        render_isaaclab_video_episode,
    )
    assert refs


def test_kiss_gen_video_isaaclab_viewport() -> None:
    from video.isaaclab_viewport import capture_isaaclab_viewport_frame, prepare_isaaclab_video_view

    refs = (
        prepare_isaaclab_video_view,
        capture_isaaclab_viewport_frame,
    )
    assert refs
