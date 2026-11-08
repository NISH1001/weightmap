"""Cross-tokenizer token alignment by shared character boundaries.

Two tokens from different tokenizers that *end at the same character* of the same text sit at
positions where both causal models have read exactly the same text prefix, so their hidden states
are comparable. With identical tokenizers every token boundary is shared, so this reduces to
ordinary position-by-position pairing.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def token_ends(offsets: torch.Tensor, mask: torch.Tensor) -> dict[int, int]:
    """Map character end-offset -> token index for real (non-pad, non-zero-width) tokens of one row.
    The first real token is skipped (attention sink). If several tokens share an end offset
    (byte-level pieces of one character), the last one - which has seen the whole character - wins."""
    ends: dict[int, int] = {}
    first = True
    for i in range(len(mask)):
        if not mask[i]:
            continue
        s, e = int(offsets[i, 0]), int(offsets[i, 1])
        if first:
            first = False
            continue
        if e <= s:                  # zero-width: special tokens added by the tokenizer
            continue
        ends[e] = i
    return ends


def boundary_pairs(offs_a: torch.Tensor, mask_a: torch.Tensor,
                   offs_b: torch.Tensor, mask_b: torch.Tensor) -> list[tuple[int, int]]:
    """(index_in_a, index_in_b) pairs whose tokens end at the same character, in text order."""
    ea, eb = token_ends(offs_a, mask_a), token_ends(offs_b, mask_b)
    return [(ea[e], eb[e]) for e in sorted(ea.keys() & eb.keys())]


def positions_from_char(offsets: torch.Tensor, mask: torch.Tensor, start_char: int) -> list[int]:
    """Indices of real tokens that start at or after `start_char` (e.g. the completion after a prompt)."""
    return [i for i in range(len(mask))
            if mask[i] and int(offsets[i, 1]) > int(offsets[i, 0]) and int(offsets[i, 0]) >= start_char]


def tokenize(tok, texts: Sequence[str], max_length: int | None = None):
    """Left-padded batch with character offsets. Returns (model_inputs, offsets)."""
    if not getattr(tok, "is_fast", False):
        raise ValueError(f"{type(tok).__name__} is not a fast tokenizer; character offsets are required")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        kw = dict(truncation=True, max_length=max_length) if max_length else {}
        enc = tok(list(texts), return_tensors="pt", padding=True, return_offsets_mapping=True, **kw)
    finally:
        tok.padding_side = side
    offsets = enc.pop("offset_mapping")
    return enc, offsets
