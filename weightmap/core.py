"""WeightMap: discover a training-free linear map between a large and a small LM's representations.

The map lives in *activation space*: for each small-model layer s it pairs the best-matching
large-model layer l (by linear CKA) and fits closed-form ridge maps h_L(l) -> h_S(s) and back,
on residual-stream states at aligned tokens. Tokens are aligned by shared character boundaries,
so the two models may use different tokenizers (identical tokenizers align every position).
Nothing is trained.

The map can then be used to
  * translate states / directions between the models (`to_small`, `to_large`, `transport`),
  * build correct-vs-wrong style contrast directions (`contrast_direction`),
  * steer the small model (`steer`), or transplant the large model's mapped prompt states into it
    (`generate(..., transplant_layer=...)`),
  * fold a static steering vector into the small model's weights (`fold`) - the one true weight edit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from . import align, linalg
from .models import (ModelSpec, decoder_layers, empty_cache, hidden_states, hook_layer_output,
                     model_device, resolve)


@dataclass
class LayerMap:
    small_layer: int
    large_layer: int
    large_to_small: linalg.Ridge
    small_to_large: linalg.Ridge
    stats: dict = field(default_factory=dict)


class FoldHandle:
    """Returned by `WeightMap.fold`; call `.undo()` to restore the original weights."""

    def __init__(self, undo_fn):
        self._undo = undo_fn
        self.active = True

    def undo(self) -> None:
        if self.active:
            self._undo()
            self.active = False


class WeightMap:
    """Training-free map between a large and a small causal LM.

    large / small: a hub id or local path, a loaded `PreTrainedModel`, or a `(model, tokenizer)` pair.
    The tokenizers may differ (both must be "fast" tokenizers that report character offsets).

    >>> wm = WeightMap(large="Qwen/Qwen2.5-1.5B-Instruct", small="Qwen/Qwen2.5-0.5B-Instruct")
    >>> wm.fit(texts)                       # CKA layer matching + ridge maps, closed form
    >>> wm.report()                         # R^2 both ways, shuffled control, reference
    >>> d = wm.contrast_direction(good, bad, layer=10, which="large")
    >>> with wm.steer(wm.transport(d, 10), layer=10): ...
    """

    def __init__(self, large: ModelSpec, small: ModelSpec, tokenizer: PreTrainedTokenizerBase | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        self.large, self.tok_large = resolve(large, tokenizer, device, dtype)
        self.small, self.tok_small = resolve(small, tokenizer, device, dtype)
        for t in (self.tok_large, self.tok_small):
            if t.pad_token is None:
                t.pad_token = t.eos_token
        self.shared_vocab = self.tok_large.get_vocab() == self.tok_small.get_vocab()
        self.large_name = _name(large, self.large)
        self.small_name = _name(small, self.small)
        # last hidden state has the final norm applied -> excluded from matching
        self.n_small = len(decoder_layers(self.small)) - 1
        self.n_large = len(decoder_layers(self.large)) - 1
        self.cka: torch.Tensor | None = None
        self.layer_match: dict[int, int] = {}
        self.maps: dict[int, LayerMap] = {}
        self._dev_cache: dict = {}

    # ------------------------------------------------------------------ discovery
    def fit(self, texts: Sequence[str], layers: Iterable[int] | None = None, positions_per_text: int = 32,
            batch_size: int = 4, max_length: int = 1024, cka_texts: int = 64,
            ridge_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1), holdout: float = 0.2,
            outlier_factor: float = 5.0, max_tokens: int = 50_000, seed: int = 0) -> "WeightMap":
        """Discover the layer correspondence (linear CKA) and fit ridge maps in both directions.

        texts: any text both models can read; in-domain text gives better maps.
        layers: small-model layers to fit maps for (default: all except the final one).
        """
        texts = list(texts)
        all_s, all_l = list(range(self.n_small)), list(range(self.n_large))
        # pass 1: all-pairs CKA on a subset of texts
        XS, XL = self._collect(texts[:cka_texts], all_s, all_l, positions_per_text, batch_size, max_length, seed)
        keep = _inlier_mask(XS[self.n_small // 2], XL[self.n_large // 2], outlier_factor)
        self.cka = torch.tensor([[linalg.linear_cka(XS[a][keep], XL[b][keep]) for b in all_l] for a in all_s])
        self.layer_match = {a: int(self.cka[a].argmax()) for a in all_s}
        del XS, XL
        # pass 2: selected layer pairs over all texts
        layers = sorted(set(all_s if layers is None else layers))
        need_s = sorted(set(layers) | {a - 1 for a in layers if a > 0})
        need_l = sorted({self.layer_match[a] for a in layers})
        XS, XL = self._collect(texts, need_s, need_l, positions_per_text, batch_size, max_length, seed)
        gen = torch.Generator().manual_seed(seed)
        self.maps = {}
        for a in layers:
            b = self.layer_match[a]
            xs, xl = XS[a].double(), XL[b].double()
            keep = _inlier_mask(xs, xl, outlier_factor)
            idx = keep.nonzero().squeeze(-1)
            idx = idx[torch.randperm(len(idx), generator=gen)][:max_tokens]
            xs, xl = xs[idx], xl[idx]
            ls, r2_ls = linalg.fit_ridge_cv(xl, xs, ridge_grid, holdout, seed)
            sl, r2_sl = linalg.fit_ridge_cv(xs, xl, ridge_grid, holdout, seed)
            shuffled = xl[torch.randperm(len(xl), generator=gen)]
            r2_shuf = _shuffled_r2(shuffled, xl, xs, ls.lam, holdout, seed)
            stats = {"small_layer": a, "large_layer": b, "cka": float(self.cka[a, b]),
                     "r2_large_to_small": r2_ls, "r2_small_to_large": r2_sl,
                     "r2_shuffled_control": r2_shuf, "n_tokens": len(idx)}
            if a > 0:
                prev = XS[a - 1].double()[idx]
                stats["r2_small_prev_layer"] = linalg.heldout_r2(prev, xs, 1e-3, holdout, seed)
            else:
                stats["r2_small_prev_layer"] = float("nan")
            self.maps[a] = LayerMap(a, b, ls, sl, stats)
        self._dev_cache.clear()
        return self

    def report(self) -> list[dict]:
        """One row per fitted layer: CKA, held-out R^2 both directions, shuffled control, reference."""
        return [dict(m.stats) for _, m in sorted(self.maps.items())]

    @property
    def mapped_layers(self) -> list[int]:
        return sorted(self.maps)

    # ------------------------------------------------------------------ translation
    def to_small(self, h_large: torch.Tensor, layer: int) -> torch.Tensor:
        """Map large-model states at the large layer paired with small `layer` into small space."""
        return self._apply(h_large, self._map(layer).large_to_small)

    def to_large(self, h_small: torch.Tensor, layer: int) -> torch.Tensor:
        return self._apply(h_small, self._map(layer).small_to_large)

    def transport(self, direction_large: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a *direction* (difference vector) from large to small space: linear part only, no bias."""
        W = self._map(layer).large_to_small.W
        return (direction_large.double().cpu() @ W).float()

    # ------------------------------------------------------------------ directions
    @torch.no_grad()
    def encode(self, texts: Sequence[str], layer: int, which: str = "small",
               start_chars: Sequence[int] | None = None, batch_size: int = 4,
               max_length: int = 2048) -> torch.Tensor:
        """Per-text mean residual state at `layer`.
        start_chars: per text, only average tokens starting at/after this character (e.g. len(prompt)
        to average over a completion). Character-based, so it means the same thing for any tokenizer.
        Default: every real token except the first."""
        model, tok = self._which(which)
        dev = model_device(model)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc, offs = align.tokenize(tok, chunk, max_length)
            hs = hidden_states(model, {k: v.to(dev) for k, v in enc.items()})[layer]
            for j in range(len(chunk)):
                mask = enc["attention_mask"][j].bool()
                if start_chars is not None:
                    pos = align.positions_from_char(offs[j], mask, start_chars[i + j])
                else:
                    pos = mask.nonzero().squeeze(-1)[1:].tolist()
                if pos:
                    out.append(hs[j, pos].float().mean(0).cpu())
            del hs, enc
            empty_cache(dev)
        return torch.stack(out)

    def contrast_direction(self, positive: Sequence[str], negative: Sequence[str], layer: int,
                           which: str = "small", start_chars: Sequence[int] | None = None,
                           **kw) -> torch.Tensor:
        """mean(positive) - mean(negative) in the chosen model's space at `layer`.
        For which="large", `layer` is the *large* layer index; use `transport` to bring it to small space.
        start_chars applies to both lists (same prompts, different completions)."""
        pos = self.encode(positive, layer, which, start_chars, **kw).mean(0)
        neg = self.encode(negative, layer, which, start_chars, **kw).mean(0)
        return pos - neg

    # ------------------------------------------------------------------ editing the small model
    def steer(self, vector: torch.Tensor, layer: int):
        """Context manager: add `vector` to the small model's residual stream after `layer`, all positions."""
        return hook_layer_output(self.small, layer, lambda h: h + vector.to(device=h.device, dtype=h.dtype))

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 256, transplant_layer: int | None = None,
                 beta: float = 0.5, steer: tuple[torch.Tensor, int] | None = None, batch_size: int = 16,
                 outlier_factor: float = 5.0, **gen_kwargs) -> list[str]:
        """Greedy generation with the small model, optionally
          * transplanting the large model's mapped prompt states at small `transplant_layer`:
            h <- (1 - beta) * h + beta * to_small(h_large)   (prefill only; attention-sink tokens untouched)
          * steering with `steer=(vector, layer)`.
        Prompts are used as-is (apply a chat template yourself if needed)."""
        gen_kwargs.setdefault("do_sample", False)
        dev = model_device(self.small)
        outs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = list(prompts[i:i + batch_size])
            enc, offs = align.tokenize(self.tok_small, chunk)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with self._transplant_ctx(chunk, enc, offs, transplant_layer, beta, outlier_factor), \
                    self._steer_ctx(steer):
                g = self.small.generate(**enc, max_new_tokens=max_new_tokens,
                                        pad_token_id=self.tok_small.pad_token_id, **gen_kwargs)
            outs += self.tok_small.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            del g, enc
            empty_cache(dev)
        return outs

    @torch.no_grad()
    def fold(self, vector: torch.Tensor, layer: int) -> FoldHandle:
        """Fold a static steering vector into the small model's weights as a bias on layer `layer`'s MLP
        output projection. Exactly equivalent to `steer(vector, layer)` (up to float rounding order).
        Returns a handle whose `.undo()` restores the original module."""
        mlp = getattr(decoder_layers(self.small)[layer], "mlp", None)
        down = getattr(mlp, "down_proj", None)
        if not isinstance(down, torch.nn.Linear):
            raise NotImplementedError("fold() needs a Llama/Qwen-style layer.mlp.down_proj nn.Linear")
        v = vector.to(device=down.weight.device, dtype=down.weight.dtype)
        if down.bias is not None:
            down.bias.add_(v)
            return FoldHandle(lambda: down.bias.sub_(v))
        new = torch.nn.Linear(down.in_features, down.out_features, bias=True,
                              device=down.weight.device, dtype=down.weight.dtype)
        new.weight = down.weight            # share, don't copy
        new.bias.data.copy_(v)
        mlp.down_proj = new
        return FoldHandle(lambda: setattr(mlp, "down_proj", down))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        torch.save({
            "version": 1, "large": self.large_name, "small": self.small_name,
            "cka": self.cka, "layer_match": self.layer_match,
            "maps": {a: {"large_layer": m.large_layer, "stats": m.stats,
                         "ls": (m.large_to_small.W, m.large_to_small.b, m.large_to_small.lam),
                         "sl": (m.small_to_large.W, m.small_to_large.b, m.small_to_large.lam)}
                     for a, m in self.maps.items()},
        }, path)

    @classmethod
    def load(cls, path: str | Path, large: ModelSpec, small: ModelSpec, **kw) -> "WeightMap":
        state = torch.load(path, weights_only=False)
        wm = cls(large=large, small=small, **kw)
        wm.cka, wm.layer_match = state["cka"], {int(k): v for k, v in state["layer_match"].items()}
        wm.maps = {int(a): LayerMap(int(a), d["large_layer"], linalg.Ridge(*d["ls"]), linalg.Ridge(*d["sl"]), d["stats"])
                   for a, d in state["maps"].items()}
        return wm

    # ------------------------------------------------------------------ internals
    def _which(self, which: str):
        if which not in ("small", "large"):
            raise ValueError("which must be 'small' or 'large'")
        return (self.small, self.tok_small) if which == "small" else (self.large, self.tok_large)

    def _map(self, layer: int) -> LayerMap:
        if not self.maps:
            raise RuntimeError("no maps yet: call fit() first (or WeightMap.load)")
        if layer not in self.maps:
            raise KeyError(f"no map for small layer {layer}; fitted layers: {self.mapped_layers}")
        return self.maps[layer]

    @staticmethod
    def _apply(h: torch.Tensor, r: linalg.Ridge) -> torch.Tensor:
        shape = h.shape
        y = r(h.reshape(-1, shape[-1]).double().cpu()).float()
        return y.reshape(*shape[:-1], y.shape[-1]).to(h.device)

    @torch.no_grad()
    def _collect(self, texts, s_layers, l_layers, k, batch_size, max_length, seed):
        """Residual states of both models at aligned tokens (shared character boundaries; k random
        aligned pairs per text, first token excluded). Returns ({s: [N, dS]}, {l: [N, dL]}), fp16 on CPU."""
        XS = {a: [] for a in s_layers}
        XL = {b: [] for b in l_layers}
        dev_s, dev_l = model_device(self.small), model_device(self.large)
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc_s, offs_s = align.tokenize(self.tok_small, chunk, max_length)
            enc_l, offs_l = align.tokenize(self.tok_large, chunk, max_length)
            picks_s, picks_l = [], []
            for j in range(len(chunk)):
                pairs = align.boundary_pairs(offs_s[j], enc_s["attention_mask"][j].bool(),
                                             offs_l[j], enc_l["attention_mask"][j].bool())
                if len(pairs) > k:
                    g = torch.Generator().manual_seed(seed * 1_000_003 + i + j)
                    sel = torch.randperm(len(pairs), generator=g)[:k].sort().values.tolist()
                    pairs = [pairs[t] for t in sel]
                picks_s.append(torch.tensor([p[0] for p in pairs], dtype=torch.long))
                picks_l.append(torch.tensor([p[1] for p in pairs], dtype=torch.long))
            for model, dev, enc, picks, layers, store in (
                    (self.small, dev_s, enc_s, picks_s, s_layers, XS),
                    (self.large, dev_l, enc_l, picks_l, l_layers, XL)):
                hs = hidden_states(model, {kk: v.to(dev) for kk, v in enc.items()})
                for j, pos in enumerate(picks):
                    pos = pos.to(dev)
                    for a in layers:
                        store[a].append(hs[a][j, pos].to(torch.float16).cpu())
                del hs
                empty_cache(dev)
        return ({a: torch.cat(v) for a, v in XS.items()}, {b: torch.cat(v) for b, v in XL.items()})

    @contextmanager
    def _transplant_ctx(self, prompts, enc_s, offs_s, layer, beta, outlier_factor):
        if layer is None:
            yield
            return
        m = self._map(layer)
        r = m.large_to_small
        dev_s = model_device(self.small)
        key = (layer, str(dev_s))
        if key not in self._dev_cache:
            self._dev_cache[key] = (r.W.float().to(dev_s), r.b.float().to(dev_s))
        W, b = self._dev_cache[key]
        enc_l, offs_l = align.tokenize(self.tok_large, prompts)
        dev_l = model_device(self.large)
        hL = hidden_states(self.large, {k: v.to(dev_l) for k, v in enc_l.items()})[m.large_layer]
        mapped = hL.float().to(dev_s) @ W + b                       # [B, T_large, d_small]
        B, plen = enc_s["input_ids"].shape
        target = torch.zeros(B, plen, W.shape[1], device=dev_s)
        aligned = torch.zeros(B, plen, 1, dtype=torch.bool, device=dev_s)
        mask_s, mask_l = enc_s["attention_mask"].bool().cpu(), enc_l["attention_mask"].bool()
        for j in range(B):
            for i_s, i_l in align.boundary_pairs(offs_s[j], mask_s[j], offs_l[j], mask_l[j]):
                target[j, i_s] = mapped[j, i_l]
                aligned[j, i_s] = True

        def fn(h):
            if h.shape[1] != plen:          # decoding steps: untouched
                return h
            hf = h.float()
            n = hf.norm(dim=-1, keepdim=True)
            sink = n > outlier_factor * n.median()
            mixed = (1 - beta) * hf + beta * target
            return torch.where(aligned & ~sink, mixed, hf).to(h.dtype)

        with hook_layer_output(self.small, layer, fn):
            yield

    @contextmanager
    def _steer_ctx(self, steer):
        if steer is None:
            yield
            return
        vec, layer = steer
        with self.steer(vec, layer):
            yield


def _name(spec, model) -> str:
    if isinstance(spec, str):
        return spec
    return getattr(getattr(model, "config", None), "_name_or_path", "") or type(model).__name__


def _inlier_mask(xs: torch.Tensor, xl: torch.Tensor, factor: float) -> torch.Tensor:
    """Drop attention-sink / massive-activation tokens (norm > factor x median in either model)."""
    ns, nl = xs.float().norm(dim=1), xl.float().norm(dim=1)
    return (ns < factor * ns.median()) & (nl < factor * nl.median())


def _shuffled_r2(x_shuffled, x_true, y, lam, holdout, seed) -> float:
    """Fit on shuffled (broken) pairs, evaluate on true held-out pairs: is token correspondence needed?"""
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    ntr = int((1 - holdout) * n)
    tr, te = perm[:ntr], perm[ntr:]
    m = linalg.fit_ridge(x_shuffled[tr], y[tr], lam)
    return linalg.r2(m, x_true[te], y[te])
