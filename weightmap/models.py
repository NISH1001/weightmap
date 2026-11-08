"""Model resolution and architecture access for HF decoder-only causal LMs."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

ModelSpec = str | PreTrainedModel | tuple[PreTrainedModel, PreTrainedTokenizerBase]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve(spec: ModelSpec, tokenizer: PreTrainedTokenizerBase | None, device: str | None,
            dtype: torch.dtype | None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Accept a hub id / local path, a loaded model, or a (model, tokenizer) pair."""
    if isinstance(spec, tuple):
        model, tok = spec
    elif isinstance(spec, str):
        dev = device or default_device()
        dt = dtype or (torch.float32 if dev == "cpu" else torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(spec, dtype=dt).to(dev)
        tok = tokenizer or AutoTokenizer.from_pretrained(spec)
    elif isinstance(spec, nn.Module):
        model = spec
        tok = tokenizer
        if tok is None:
            name = getattr(getattr(model, "config", None), "_name_or_path", None)
            if not name:
                raise ValueError("pass a tokenizer for an in-memory model, e.g. small=(model, tokenizer)")
            tok = AutoTokenizer.from_pretrained(name)
    else:
        raise TypeError(f"unsupported model spec: {type(spec)!r}")
    model.eval()
    return model, tok


def base_model(model: PreTrainedModel) -> nn.Module:
    """The transformer body (no LM head), so hidden states don't materialise [B, T, vocab] logits."""
    return model.base_model if model.base_model is not model else model


def decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    body = base_model(model)
    for name in ("layers", "h", "blocks"):
        layers = getattr(body, name, None)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise AttributeError("could not find the decoder layer list (looked for .layers / .h / .blocks)")


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def hidden_states(model: PreTrainedModel, enc: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Residual stream after each decoder layer: index l = output of layer l.
    (The last entry has the final norm applied in HF models; callers usually exclude it.)"""
    out = base_model(model)(**enc, output_hidden_states=True)
    return out.hidden_states[1:]


@contextmanager
def hook_layer_output(model: PreTrainedModel, layer: int, fn):
    """Replace the residual-stream output h of decoder layer `layer` by fn(h) for the duration."""
    def hook(_m, _i, out: Any):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
