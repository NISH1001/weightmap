"""Calibration controls: how much of the map's R^2 is just "both models saw the same token"?

For S in {Qwen2.5-0.5B (same tokenizer), SmolLM2-135M (other family)} and S-layers at ~40% / ~60% depth,
predict S's state from:
  trained L, best layer   - the real map (layer chosen on an inner split, like every condition)
  trained L, embeddings   - L's input embedding only: pure token identity in L's coordinates
  random-init L, best layer - same architecture/tokenizer, untrained: token identity + random mixing of context
  random-init L, embeddings
  S's own embeddings      - how much of S's mid-layer its own token embedding already explains
and the reverse direction (S-layer -> trained/random L, best layer).
Same texts, same aligned positions, text-level holdout. No training: forward passes + closed-form ridge.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/controls_baselines.py
"""
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weightmap import align, linalg
from weightmap.models import base_model, default_device, empty_cache

HERE = Path(__file__).parent
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
OUT = HERE / "controls_baselines_results.json"
LARGE = "Qwen/Qwen2.5-1.5B-Instruct"
SMALL = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M"]
GRID = (1e-3, 1e-2, 1e-1, 1.0)
K, BS, MAXLEN, CAP = 32, 4, 512, 12000
DEV = default_device()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
texts = [t for t in texts if len(t.strip()) > 20]
log(len(texts), "texts")


def load(name, random_init=False):
    tok = AutoTokenizer.from_pretrained(name)
    if random_init:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    return model.to(DEV).eval(), tok


@torch.no_grad()
def all_states(model, enc):
    """(embedding output, layer 0 out, ..., last layer out) - index 0 is the embedding."""
    return base_model(model)(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True).hidden_states


@torch.no_grad()
def collect(S, tokS, Ls, tokL, s_layers):
    """S states at s_layers + S embedding; every layer (incl. embedding) of each model in Ls (same tokenizer)."""
    xs = {a: [] for a in s_layers + ["emb"]}
    xl = {name: [] for name in Ls}
    groups = []
    for i in range(0, len(texts), BS):
        chunk = texts[i:i + BS]
        enc_s, off_s = align.tokenize(tokS, chunk, MAXLEN)
        enc_l, off_l = align.tokenize(tokL, chunk, MAXLEN)
        ps, pl = [], []
        for j in range(len(chunk)):
            pairs = align.boundary_pairs(off_s[j], enc_s["attention_mask"][j].bool(),
                                         off_l[j], enc_l["attention_mask"][j].bool())
            if len(pairs) > K:
                g = torch.Generator().manual_seed(i + j)
                pairs = [pairs[t] for t in torch.randperm(len(pairs), generator=g)[:K].sort().values.tolist()]
            ps.append(torch.tensor([p[0] for p in pairs])); pl.append(torch.tensor([p[1] for p in pairs]))
            groups.append(torch.full((len(pairs),), i + j))
        hs = all_states(S, enc_s)
        for j, pos in enumerate(ps):
            for a in s_layers:
                xs[a].append(hs[a + 1][j, pos.to(DEV)].half().cpu())     # +1: hidden_states[0] is the embedding
            xs["emb"].append(hs[0][j, pos.to(DEV)].half().cpu())
        del hs
        for name, L in Ls.items():
            hs = all_states(L, enc_l)
            st = torch.stack(hs[:-1], 1)                                  # drop final-norm output: [B, nL, T, d]
            for j, pos in enumerate(pl):
                xl[name].append(st[j][:, pos.to(DEV)].half().cpu())       # [nL, k, d]
            del hs, st
        empty_cache(torch.device(DEV))
    return ({a: torch.cat(v).float() for a, v in xs.items()},
            {n: torch.cat(v, 1).float() for n, v in xl.items()}, torch.cat(groups))


def heldout(X, Y, tr_in, va_in, tr, te):
    """lambda on inner split of training texts, refit on train, R^2 on held-out texts. Returns (val, test)."""
    best = max(((linalg.heldout_r2(X, Y, lam, tr_in, va_in), lam) for lam in GRID))
    return best[0], linalg.heldout_r2(X, Y, best[1], tr, te)


def best_layer(Xs_by_layer, Y, splits, reverse=False):
    """Pick the layer on the inner validation split; report its test R^2."""
    rows = []
    for l, X in enumerate(Xs_by_layer):
        v, t = heldout(Y, X, *splits) if reverse else heldout(X, Y, *splits)
        rows.append((v, t, l))
    v, t, l = max(rows)
    return {"layer": l, "r2": round(t, 4), "val": round(v, 4)}


res = {"setup": {"large": LARGE, "k": K, "max_length": MAXLEN, "cap_tokens": CAP, "grid": GRID}, "pairs": []}
Lt, tokL = load(LARGE)
Lr, _ = load(LARGE, random_init=True)
log("loaded trained + random-init L")
for name in SMALL:
    t0 = time.time()
    S, tokS = load(name)
    n = len(base_model(S).layers) - 1
    s_layers = sorted({round(0.4 * n), round(0.6 * n)})
    XS, XL, groups = collect(S, tokS, {"trained": Lt, "random": Lr}, tokL, s_layers)
    # outliers (attention-sink-like tokens) on the trained models' mid layers, as the library does
    mS, mL = XS[s_layers[0]].norm(dim=1), XL["trained"][XL["trained"].shape[0] // 2].norm(dim=1)
    keep = ((mS < 5 * mS.median()) & (mL < 5 * mL.median())).nonzero().squeeze(-1)
    keep = keep[torch.randperm(len(keep), generator=torch.Generator().manual_seed(0))][:CAP]
    XS = {a: v[keep] for a, v in XS.items()}
    XL = {k_: v[:, keep] for k_, v in XL.items()}
    groups = groups[keep]
    tr, te = linalg.split(len(groups), 0.2, 0, groups)
    tr_in, va_in = (tr[t] for t in linalg.split(len(tr), 0.2, 1, groups[tr]))
    splits = (tr_in, va_in, tr, te)
    entry = {"small": name, "n_tokens": len(groups), "n_test": len(te), "layers": []}
    for s in s_layers:
        Y = XS[s]
        row = {"small_layer": s,
               "L_to_S": {"trained_best": best_layer(XL["trained"], Y, splits),
                          "trained_embedding": round(heldout(XL["trained"][0], Y, *splits)[1], 4),
                          "random_best": best_layer(XL["random"], Y, splits),
                          "random_embedding": round(heldout(XL["random"][0], Y, *splits)[1], 4),
                          "S_own_embedding": round(heldout(XS["emb"], Y, *splits)[1], 4)},
               "S_to_L": {"trained_best": best_layer(XL["trained"], Y, splits, reverse=True),
                          "random_best": best_layer(XL["random"], Y, splits, reverse=True)}}
        entry["layers"].append(row)
        log(name, "S layer", s, json.dumps(row))
    res["pairs"].append(entry)
    log(f"done {name} [{time.time() - t0:.0f}s]")
    OUT.write_text(json.dumps(res, indent=1))
    del S, XS, XL
    gc.collect(); empty_cache(torch.device(DEV))
log("wrote", OUT)
