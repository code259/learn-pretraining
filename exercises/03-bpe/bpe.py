"""Lesson 03: byte-level BPE from scratch. Fill in every `raise NotImplementedError`.
Run `python test_bpe.py` after each function. Pure Python, no dependencies.

Representation: a "word" (pre-token chunk) is a tuple of bytes objects,
e.g. " cat" -> (b" ", b"c", b"a", b"t"). Merging (b"a", b"t") gives (b" ", b"c", b"at").
"""
import re
from collections import Counter

# Simplified ASCII version of the GPT-2 pre-tokenizer regex, plus SmolLM2's
# "split every digit" rule. (The real one uses the `regex` module's \p{L}/\p{N}.)
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-zÀ-￿]+| ?[0-9]|[0-9]| ?[^\sA-Za-z0-9À-￿]+|\s+(?!\S)|\s+""")


def pretokenize(text: str) -> list[str]:
    """Split text into chunks. Merges never cross chunk boundaries."""
    chunks = PAT.findall(text)
    # SmolLM2 splits digits individually *before* the byte-level regex, so " 2" -> " ", "2".
    out = []
    for c in chunks:
        if len(c) == 2 and c[0] == " " and c[1].isdigit():
            out += [" ", c[1]]
        else:
            out.append(c)
    return out


def get_pair_counts(words: dict[tuple[bytes, ...], int]) -> Counter:
    """Count adjacent symbol pairs, weighted by how often each word occurs."""
    raise NotImplementedError


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Replace every non-overlapping occurrence of `pair` (scanning left to right) with the joined symbol."""
    raise NotImplementedError


def train_from_counts(word_counts: dict[str, int], num_merges: int) -> list[tuple[bytes, bytes]]:
    """The core BPE loop. Start from bytes; repeat: count pairs, pick the most frequent
    (ties -> lexicographically GREATER pair, e.g. (b"s", b"t") beats (b"e", b"s")),
    merge it in every word, record it. Return the merges in the order learned."""
    raise NotImplementedError


def train_bpe(text: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Pretokenize, count chunks, call train_from_counts for (vocab_size - 256) merges.
    Return vocab {id: bytes} where ids 0..255 are single bytes and id 256+i is merge i."""
    raise NotImplementedError


def encode(text: str, merges: list[tuple[bytes, bytes]]) -> list[int]:
    """Pretokenize, then for each chunk: start from bytes and repeatedly apply the
    applicable merge with the LOWEST rank (earliest learned) until none applies.
    Map the final symbols to ids (byte b -> b; merge i -> 256 + i)."""
    raise NotImplementedError


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Concatenate the bytes, then decode as UTF-8."""
    raise NotImplementedError
