"""Estimativas reprodutíveis; cl100k_base não é o tokenizer nativo do Gemma."""

import hashlib
from functools import lru_cache

import tiktoken
from langchain_core.messages import BaseMessage
from tiktoken.load import load_tiktoken_bpe

from app.config import ROOT

ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def encoding():
    path = ROOT / "data" / "tokenizer" / "cl100k_base.tiktoken"
    expected = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Vocabulário local do tiktoken ausente ou corrompido.")
    return tiktoken.Encoding(
        name=ENCODING,
        pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        mergeable_ranks=load_tiktoken_bpe(str(path)),
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )


def token_ids(text: str) -> list[int]:
    return encoding().encode(text, disallowed_special=())


def count_text(text: str) -> int:
    return len(token_ids(text))


def count_messages(messages: list[BaseMessage]) -> int:
    # A margem por mensagem representa papéis/delimitadores, não custo faturado.
    return sum(count_text(str(message.content)) + 4 for message in messages) + (
        2 if messages else 0
    )
