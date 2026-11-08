"""Model atlas + recursion. No training anywhere: forward passes and closed-form ridge only.

ATLAS (H1, convergence): map each small model onto Qwen2.5-1.5B (L) with WeightMap and relate map
    quality to the small model's capability (bits-per-byte on the same texts, tokenizer-agnostic).
RECURSION (H5, composition): chain Qwen-1.5B -> Qwen-0.5B -> SmolLM2-360M -> SmolLM2-135M at matched
    relative depth. Does composing the step maps predict the end model as well as a direct map?

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/atlas_recursion.py
"""
import gc
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightmap import WeightMap, align, linalg
from weightmap.models import default_device, empty_cache, hidden_states

HERE = Path(__file__).parent
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
OUT = HERE / "atlas_recursion_results.json"
LARGE = "Qwen/Qwen2.5-1.5B-Instruct"
SMALL = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-135M",
         "google/gemma-3-270m", "EleutherAI/pythia-160m"]
GRID = (1e-4, 1e-3, 1e-2, 1e-1)
DEV = default_device()
torch.manual_seed(0)
res = {"atlas": [], "recursion": {}}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    return model, tok


texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
texts = [t for t in texts if len(t.strip()) > 20]
log(len(texts), "texts")


@torch.no_grad()
def bits_per_byte(model, tok, texts, bs=2, max_length=512):
    """Capability proxy: total next-token NLL in bits / total UTF-8 bytes covered (tokenizer-agnostic)."""
    nll, nbytes = 0.0, 0
    for i in range(0, len(texts), bs):
        enc, offs = align.tokenize(tok, texts[i:i + bs], max_length)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        logits = model(**enc).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = enc["input_ids"][:, 1:]
        m = enc["attention_mask"][:, 1:].bool() & enc["attention_mask"][:, :-1].bool()
        nll -= float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[m].sum())
        for j, t in enumerate(texts[i:i + bs]):
            real = enc["attention_mask"][j].bool().cpu()
            o = offs[j][real]
            covered = t[int(o[1:, 0].min()):int(o[:, 1].max())] if len(o) > 1 else ""
            nbytes += len(covered.encode("utf-8"))
        del logits, lp
        empty_cache(torch.device(DEV))
    return nll / math.log(2) / max(nbytes, 1)


# ============================================================ ATLAS
L, tokL = load(LARGE)
bpb_L = bits_per_byte(L, tokL, texts[:200])
log("L bits/byte", round(bpb_L, 4))
res["large"] = {"name": LARGE, "bits_per_byte": round(bpb_L, 4)}
for name in SMALL:
    t0 = time.time()
    S, tokS = load(name)
    bpb = bits_per_byte(S, tokS, texts[:200])
    wm = WeightMap(large=(L, tokL), small=(S, tokS))
    layers = sorted({round(0.4 * wm.n_small), round(0.6 * wm.n_small)})
    wm.fit(texts, layers=layers, positions_per_text=32, batch_size=4, max_length=512, cka_texts=80,
           max_tokens=12000)
    rows = wm.report()
    entry = {"small": name, "n_layers": wm.n_small + 1, "shared_vocab": wm.shared_vocab,
             "params_M": round(sum(p.numel() for p in S.parameters()) / 1e6, 1),
             "bits_per_byte": round(bpb, 4),
             "mean_r2_L_to_S": round(sum(r["r2_large_to_small"] for r in rows) / len(rows), 4),
             "mean_r2_S_to_L": round(sum(r["r2_small_to_large"] for r in rows) / len(rows), 4),
             "layers": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}
    res["atlas"].append(entry)
    log(f"ATLAS {name} [{time.time() - t0:.0f}s]",
        {k: entry[k] for k in ("params_M", "bits_per_byte", "mean_r2_L_to_S", "mean_r2_S_to_L", "shared_vocab")})
    del wm, S
    gc.collect(); empty_cache(torch.device(DEV))
    OUT.write_text(json.dumps(res, indent=1))

# ============================================================ RECURSION
CHAIN = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M",
         "HuggingFaceTB/SmolLM2-135M"]
models = {LARGE: (L, tokL)}
for name in CHAIN[1:]:
    models[name] = load(name)


def mid_layer(model):
    n = len(model.model.layers) if hasattr(model, "model") else len(model.base_model.layers)
    return round(0.45 * (n - 1))


LAYER = {n: mid_layer(models[n][0]) for n in CHAIN}
log("chain layers (relative depth 0.45):", LAYER)
tokQ, tokSm = models[CHAIN[0]][1], models[CHAIN[2]][1]   # Qwen tokenizer (1.5B, 0.5B), SmolLM2 tokenizer (360M, 135M)
acts = {n: [] for n in CHAIN}
groups = []
K, BS = 32, 4
with torch.no_grad():
    for i in range(0, len(texts), BS):
        chunk = texts[i:i + BS]
        encQ, offQ = align.tokenize(tokQ, chunk, 512)
        encM, offM = align.tokenize(tokSm, chunk, 512)
        picks = []
        for j in range(len(chunk)):
            pairs = align.boundary_pairs(offQ[j], encQ["attention_mask"][j].bool(),
                                         offM[j], encM["attention_mask"][j].bool())
            if len(pairs) > K:
                g = torch.Generator().manual_seed(i + j)
                sel = torch.randperm(len(pairs), generator=g)[:K].sort().values.tolist()
                pairs = [pairs[t] for t in sel]
            picks.append(pairs)
            groups.append(torch.full((len(pairs),), i + j, dtype=torch.long))
        for n in CHAIN:
            model, _ = models[n]
            enc, side = (encQ, 0) if n in CHAIN[:2] else (encM, 1)
            hs = hidden_states(model, {k: v.to(DEV) for k, v in enc.items()})[LAYER[n]]
            for j, pairs in enumerate(picks):
                pos = torch.tensor([p[side] for p in pairs], dtype=torch.long, device=DEV)
                acts[n].append(hs[j, pos].float().cpu())
            del hs
        empty_cache(torch.device(DEV))
X = {n: torch.cat(v) for n, v in acts.items()}
groups = torch.cat(groups)
keep = torch.ones(len(groups), dtype=torch.bool)
for n in CHAIN:
    nm = X[n].norm(dim=1)
    keep &= nm < 5 * nm.median()
idx = keep.nonzero().squeeze(-1)
idx = idx[torch.randperm(len(idx))][:15000]
X = {n: v[idx] for n, v in X.items()}
groups = groups[idx]
tr, te = linalg.split(len(idx), 0.2, 0, groups)
tr_in, va_in = (tr[t] for t in linalg.split(len(tr), 0.2, 1, groups[tr]))   # inner split: pick lambda
log("recursion tokens: train", len(tr), "test", len(te))


def fit(Xa, Ya):
    """Closed-form ridge; lambda chosen on an inner text-level split of the training texts only,
    then refit on all training texts. Returns (W, b) as float32 tensors."""
    def solve(rows, lam):
        x, y = Xa[rows], Ya[rows]
        mx, my = x.mean(0), y.mean(0)
        xc, yc = x - mx, y - my
        G = (xc.T @ xc).double()
        W = torch.linalg.solve(G + lam * float(x.var(0).mean()) * len(x) * torch.eye(G.shape[0], dtype=torch.float64),
                               (xc.T @ yc).double()).float()
        return W, my - mx @ W
    best = None
    for lam in GRID:
        W, b = solve(tr_in, lam)
        s = r2(Xa[va_in] @ W + b, Ya[va_in])
        if best is None or s > best[0]:
            best = (s, lam)
    return solve(tr, best[1])


def r2(P, Y):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


short = {n: n.split("/")[-1] for n in CHAIN}
steps = {}
for a, b in zip(CHAIN, CHAIN[1:]):
    W, c = fit(X[a], X[b])
    steps[(a, b)] = (W, c)
    res["recursion"][f"step {short[a]} -> {short[b]}"] = round(r2(X[a][te] @ W + c, X[b][te]), 4)


def compose(path):
    P = X[path[0]][te]
    for a, b in zip(path, path[1:]):
        W, c = steps[(a, b)]
        P = P @ W + c
    return P


for i0 in range(len(CHAIN)):
    for j0 in range(i0 + 2, len(CHAIN)):
        a, b = CHAIN[i0], CHAIN[j0]
        Wd, cd = fit(X[a], X[b])
        direct = r2(X[a][te] @ Wd + cd, X[b][te])
        composed = r2(compose(CHAIN[i0:j0 + 1]), X[b][te])
        res["recursion"][f"{short[a]} -> {short[b]}"] = {
            "direct": round(direct, 4), "composed_via_chain": round(composed, 4),
            "composed_over_direct": round(composed / direct, 3) if direct > 0 else None}
res["recursion"]["layers"] = {short[n]: LAYER[n] for n in CHAIN}
res["recursion"]["n_tokens"] = {"train": len(tr), "test": len(te)}
log("RECURSION", json.dumps(res["recursion"], indent=1))
OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
