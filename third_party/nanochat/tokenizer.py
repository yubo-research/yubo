"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

from __future__ import annotations

import copy
import os
import pickle
from functools import lru_cache

from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        tokenizer.normalizer = None
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        return [w.content for w in special_tokens_map.values()]

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            ids.append(prepend if isinstance(prepend, int) else self.encode_special(prepend))
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            ids.append(append if isinstance(append, int) else self.encode_special(append))
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        assert bos is not None
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        if isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))


class ConversationRenderer:
    """Logic for rendering chat conversations into tokens."""

    def preprocess_conversation(self, conversation):
        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        assert len(messages) >= 1
        return messages

    def render_message(self, message, i, special_tokens, add_tokens, encode_fn):
        role = message["role"]
        must_be_from = "user" if i % 2 == 0 else "assistant"
        assert role == must_be_from
        content = message["content"]

        if role == "user":
            assert isinstance(content, str)
            add_tokens(special_tokens["user_start"], 0)
            add_tokens(encode_fn(content), 0)
            add_tokens(special_tokens["user_end"], 0)
        else:
            add_tokens(special_tokens["assistant_start"], 0)
            self._render_assistant_content(content, special_tokens, add_tokens, encode_fn)
            add_tokens(special_tokens["assistant_end"], 1)

    def _render_assistant_content(self, content, tokens, add_tokens, encode_fn):
        if isinstance(content, str):
            add_tokens(encode_fn(content), 1)
        elif isinstance(content, list):
            for part in content:
                v_ids = encode_fn(part["text"])
                if part["type"] == "text":
                    add_tokens(v_ids, 1)
                elif part["type"] == "python":
                    add_tokens(tokens["python_start"], 1)
                    add_tokens(v_ids, 1)
                    add_tokens(tokens["python_end"], 1)
                elif part["type"] == "python_output":
                    add_tokens(tokens["output_start"], 0)
                    add_tokens(v_ids, 0)
                    add_tokens(tokens["output_end"], 0)
                else:
                    raise ValueError(f"Unknown part type: {part['type']}")
        else:
            raise ValueError(f"Unknown content type: {type(content)}")

    def render_conversation(self, tokenizer, conversation, max_tokens=2048):
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            t_ids = [token_ids] if isinstance(token_ids, int) else token_ids
            ids.extend(t_ids)
            mask.extend([mask_val] * len(t_ids))

        messages = self.preprocess_conversation(conversation)
        s_tokens = {
            "user_start": tokenizer.encode_special("<|user_start|>"),
            "user_end": tokenizer.encode_special("<|user_end|>"),
            "assistant_start": tokenizer.encode_special("<|assistant_start|>"),
            "assistant_end": tokenizer.encode_special("<|assistant_end|>"),
            "python_start": tokenizer.encode_special("<|python_start|>"),
            "python_end": tokenizer.encode_special("<|python_end|>"),
            "output_start": tokenizer.encode_special("<|output_start|>"),
            "output_end": tokenizer.encode_special("<|output_end|>"),
        }

        add_tokens(tokenizer.get_bos_token_id(), 0)
        for i, message in enumerate(messages):
            self.render_message(message, i, s_tokens, add_tokens, tokenizer.encode)

        return ids[:max_tokens], mask[:max_tokens]


class RustBPETokenizer:
    """Light wrapper around tiktoken but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)
        self.renderer = ConversationRenderer()

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        import rustbpe
        import tiktoken

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        import tiktoken

        return cls(tiktoken.get_encoding(tiktoken_name), "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        prepend_id = _encode_special_token(self, prepend) if prepend is not None else None
        append_id = _encode_special_token(self, append) if append is not None else None

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            return _insert_special_tokens(ids, prepend_id, append_id)
        if isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            return _insert_special_tokens(ids, prepend_id, append_id)
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.enc, f)

    def render_conversation(self, conversation, max_tokens=2048):
        return self.renderer.render_conversation(self, conversation, max_tokens)

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        RED, GREEN, RESET, GRAY = "\033[91m", "\033[92m", "\033[0m", "\033[90m"
        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids


def _encode_special_token(tokenizer, token):
    return token if isinstance(token, int) else tokenizer.encode_special(token)


def _insert_special_tokens(ids, prepend_id, append_id):
    if prepend_id is not None:
        if isinstance(ids[0], list):
            for ids_row in ids:
                ids_row.insert(0, prepend_id)
        else:
            ids.insert(0, prepend_id)
    if append_id is not None:
        if isinstance(ids[0], list):
            for ids_row in ids:
                ids_row.append(append_id)
        else:
            ids.append(append_id)
    return ids


def get_tokenizer():
    from .common import get_base_dir

    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu"):
    import torch

    from .common import get_base_dir

    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "rb") as f:
        return torch.load(f, map_location=device)
