from __future__ import annotations

import threading
from pkg_resources import resource_filename

from .load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe
from .core import Encoding

_lock = threading.RLock()
ENCODINGS: dict[str, Encoding] = {}

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

pattern_string_1 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# pattern_string_1 = r"""'s|'t|'re|'ve|'m|'ll|'d|\s?[a-zA-Z_]+|\s?\d+|\s?[^\s\w\d]+|\s+(?!\S)|\s+"""

pattern_string_2 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
# pattern_string_2 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\d]?[a-zA-Z_]+|\d{1,3}| ?[^\s\w\d]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def gpt2():
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=resource_filename("tiktoken_offline", "./data/gpt2_vocab.bpe"),
        encoder_json_file=resource_filename("tiktoken_offline", "./data/gpt2_encoder.json"),
    )
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        "pat_str": pattern_string_1,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def r50k_base():
    mergeable_ranks = load_tiktoken_bpe(
        resource_filename("tiktoken_offline", "./data/r50k_base.tiktoken")
    )
    return {
        "name": "r50k_base",
        "explicit_n_vocab": 50257,
        "pat_str": pattern_string_1,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_base():
    mergeable_ranks = load_tiktoken_bpe(
        resource_filename("tiktoken_offline", "./data/p50k_base.tiktoken")
    )
    return {
        "name": "p50k_base",
        "explicit_n_vocab": 50281,
        "pat_str": pattern_string_1,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_edit():
    mergeable_ranks = load_tiktoken_bpe(
        resource_filename("tiktoken_offline", "./data/p50k_base.tiktoken")
    )
    special_tokens = {ENDOFTEXT: 50256, FIM_PREFIX: 50281, FIM_MIDDLE: 50282, FIM_SUFFIX: 50283}
    return {
        "name": "p50k_edit",
        "pat_str": pattern_string_1,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


def cl100k_base():
    mergeable_ranks = load_tiktoken_bpe(
        resource_filename("tiktoken_offline", "./data/cl100k_base.tiktoken")
    )
    special_tokens = {
        ENDOFTEXT: 100257,
        FIM_PREFIX: 100258,
        FIM_MIDDLE: 100259,
        FIM_SUFFIX: 100260,
        ENDOFPROMPT: 100276,
    }
    return {
        "name": "cl100k_base",
        "pat_str": pattern_string_2,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }

ENCODING_CONSTRUCTORS = {
    "gpt2": gpt2,
    "r50k_base": r50k_base,
    "p50k_base": p50k_base,
    "p50k_edit": p50k_edit,
    "cl100k_base": cl100k_base,
}

def get_encoding(encoding_name: str) -> Encoding:
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(
                f"Unknown encoding {encoding_name}. Available encodings: {list(ENCODING_CONSTRUCTORS)}"
            )

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODINGS[encoding_name] = enc
        return enc


def list_encoding_names() -> list[str]:
    with _lock:
        return list(ENCODING_CONSTRUCTORS)
