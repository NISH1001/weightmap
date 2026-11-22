"""Recursion (composed maps) over several seeds, to put error bars on scripts/atlas_recursion.py's single run.
Collect up to 64 aligned positions per text once; each seed re-draws 32 positions per text, the token cap,
and the text-level train/test split. Same chain, layers, ridge and lambda selection as the original.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/recursion_seeds.py
"""
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightmap import align, linalg
from weightmap.models import default_device, empty_cache, hidden_states

HERE = Path(__file__).parent
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
OUT = HERE / "recursion_seeds_results.json"
CHAIN = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M",
         "HuggingFaceTB/SmolLM2-135M"]
SEEDS = [0, 1, 2, 3, 4]
GRID = (1e-4, 1e-3, 1e-2, 1e-1)
DEV = default_device()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
texts = [t for t in texts if len(t.strip()) > 20]
models = {n: (AutoModelForCausalLM.from_pretrained(n, dtype=torch.bfloat16).to(DEV).eval(),
              AutoTokenizer.from_pretrained(n)) for n in CHAIN}
LAYER = {n: round(0.45 * (len(m.model.layers) - 1)) for n, (m, _) in models.items()}
log("layers", LAYER)
tokQ, tokSm = models[CHAIN[0]][1], models[CHAIN[2]][1]
acts = {n: [] for n in CHAIN}
groups = []
with torch.no_grad():
    for i in range(0, len(texts), 4):
        chunk = texts[i:i + 4]
        encQ, offQ = align.tokenize(tokQ, chunk, 512)
        encM, offM = align.tokenize(tokSm, chunk, 512)
        picks = []
        for j in range(len(chunk)):
            pairs = align.boundary_pairs(offQ[j], encQ["attention_mask"][j].bool(),
                                         offM[j], encM["attention_mask"][j].bool())
            if len(pairs) > 64:
                g = torch.Generator().manual_seed(i + j)
                pairs = [pairs[t] for t in torch.randperm(len(pairs), generator=g)[:64].sort().values.tolist()]
            picks.append(pairs)
            groups.append(torch.full((len(pairs),), i + j))
        for n in CHAIN:
            enc, side = (encQ, 0) if n in CHAIN[:2] else (encM, 1)
            hs = hidden_states(models[n][0], {k: v.to(DEV) for k, v in enc.items()})[LAYER[n]]
            for j, pairs in enumerate(picks):
                acts[n].append(hs[j, torch.tensor([p[side] for p in pairs], device=DEV)].float().cpu())
            del hs
        empty_cache(torch.device(DEV))
XALL = {n: torch.cat(v) for n, v in acts.items()}
GALL = torch.cat(groups)
del models
keep_all = torch.ones(len(GALL), dtype=torch.bool)
for n in CHAIN:
    nm = XALL[n].norm(dim=1)
    keep_all &= nm < 5 * nm.median()
log("collected", len(GALL), "tokens; inliers", int(keep_all.sum()))


def r2(P, Y):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def run(seed):
    g = torch.Generator().manual_seed(seed)
    # 32 positions per text (as the original), then the same 15000-token cap
    rows = []
    for t in GALL.unique():
        idx = (GALL == t).nonzero().squeeze(-1)
        rows.append(idx[torch.randperm(len(idx), generator=g)[:32]])
    idx = torch.cat(rows)
    idx = idx[keep_all[idx]]
    idx = idx[torch.randperm(len(idx), generator=g)][:15000]
    X = {n: v[idx] for n, v in XALL.items()}
    grp = GALL[idx]
    tr, te = linalg.split(len(idx), 0.2, seed, grp)
    tr_in, va_in = (tr[t] for t in linalg.split(len(tr), 0.2, seed + 100, grp[tr]))

    def fit(Xa, Ya):
        def solve(rr, lam):
            x, y = Xa[rr], Ya[rr]
            mx, my = x.mean(0), y.mean(0)
            G = ((x - mx).T @ (x - mx)).double()
            W = torch.linalg.solve(G + lam * float(x.var(0).mean()) * len(x) * torch.eye(G.shape[0], dtype=torch.float64),
                                   ((x - mx).T @ (y - my)).double()).float()
            return W, my - mx @ W
        lam = max(GRID, key=lambda l_: r2(Xa[va_in] @ solve(tr_in, l_)[0] + solve(tr_in, l_)[1], Ya[va_in]))
        return solve(tr, lam)

    out, steps = {}, {}
    for a, b in zip(CHAIN, CHAIN[1:]):
        steps[(a, b)] = fit(X[a], X[b])
        W, c = steps[(a, b)]
        out[f"step {a.split('/')[-1]} -> {b.split('/')[-1]}"] = r2(X[a][te] @ W + c, X[b][te])
    for i0 in range(len(CHAIN)):
        for j0 in range(i0 + 2, len(CHAIN)):
            a, b = CHAIN[i0], CHAIN[j0]
            Wd, cd = fit(X[a], X[b])
            direct = r2(X[a][te] @ Wd + cd, X[b][te])
            P = X[a][te]
            for u, v in zip(CHAIN[i0:j0], CHAIN[i0 + 1:j0 + 1]):
                W, c = steps[(u, v)]
                P = P @ W + c
            comp = r2(P, X[b][te])
            key = f"{a.split('/')[-1]} -> {b.split('/')[-1]}"
            out[key + " | direct"], out[key + " | composed"], out[key + " | ratio"] = direct, comp, comp / direct
    out["n_train"], out["n_test"] = len(tr), len(te)
    return out


res = {"layers": {n.split("/")[-1]: l for n, l in LAYER.items()}, "seeds": {}}
for s in SEEDS:
    res["seeds"][s] = {k: round(v, 4) for k, v in run(s).items()}
    log("seed", s, json.dumps(res["seeds"][s]))
keys = [k for k in res["seeds"][SEEDS[0]] if not k.startswith("n_")]
res["summary"] = {k: {"mean": round(statistics.mean(res["seeds"][s][k] for s in SEEDS), 4),
                      "sd": round(statistics.stdev(res["seeds"][s][k] for s in SEEDS), 4),
                      "min": round(min(res["seeds"][s][k] for s in SEEDS), 4)} for k in keys}
log("SUMMARY", json.dumps(res["summary"], indent=1))
OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
