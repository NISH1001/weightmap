"""Does a stricter map (rotation + one global scale, i.e. orthogonal Procrustes) behave like our ridge map,
and does it change how much of the fit is just token identity?

Same data, positions, outlier filter, text-level split and layer choice as scripts/controls_baselines.py.
For each S layer, predict S's state from L (Qwen2.5-1.5B-Instruct):
  trained L, best layer (chosen by ridge on the inner split) | L's embedding only | random-init L, best layer
with two maps fitted on the same training rows:
  ridge     W = (XcᵀXc + λI)⁻¹ XcᵀYc, λ on an inner split
  rotation  R = U Vᵀ from SVD(XcᵀYc) (semi-orthogonal across widths), one scale s (least squares), plus means
Held-out R² on test texts. No training.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/rotation_vs_ridge.py
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
OUT = HERE / "rotation_vs_ridge_results.json"
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
    return base_model(model)(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True).hidden_states


@torch.no_grad()
def collect(S, tokS, Ls, tokL, s_layers):
    xs = {a: [] for a in s_layers}
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
                xs[a].append(hs[a + 1][j, pos.to(DEV)].half().cpu())
        del hs
        for name, L in Ls.items():
            hs = all_states(L, enc_l)
            st = torch.stack(hs[:-1], 1)
            for j, pos in enumerate(pl):
                xl[name].append(st[j][:, pos.to(DEV)].half().cpu())
            del hs, st
        empty_cache(torch.device(DEV))
    return ({a: torch.cat(v).float() for a, v in xs.items()},
            {n: torch.cat(v, 1).float() for n, v in xl.items()}, torch.cat(groups))


def procrustes(X, Y):
    """Rotation (semi-orthogonal when widths differ) + one least-squares scale + means."""
    X, Y = X.double(), Y.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, Sv, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    XR = Xc @ R
    s = float(Sv.sum() / (XR ** 2).sum())
    return lambda Z: my + s * ((Z.double() - mx) @ R)


def r2_of(pred, Y):
    Y = Y.double()
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def both(X, Y, tr_in, va_in, tr, te):
    lam = max(GRID, key=lambda l_: linalg.heldout_r2(X, Y, l_, tr_in, va_in))
    return {"ridge": round(linalg.heldout_r2(X, Y, lam, tr, te), 4),
            "rotation": round(r2_of(procrustes(X[tr], Y[tr])(X[te]), Y[te]), 4)}


res = {"setup": {"large": LARGE, "k": K, "cap_tokens": CAP, "grid": GRID}, "pairs": []}
Lt, tokL = load(LARGE)
Lr, _ = load(LARGE, random_init=True)
for name in SMALL:
    t0 = time.time()
    S, tokS = load(name)
    n = len(base_model(S).layers) - 1
    s_layers = sorted({round(0.4 * n), round(0.6 * n)})
    XS, XL, groups = collect(S, tokS, {"trained": Lt, "random": Lr}, tokL, s_layers)
    mS, mL = XS[s_layers[0]].norm(dim=1), XL["trained"][XL["trained"].shape[0] // 2].norm(dim=1)
    keep = ((mS < 5 * mS.median()) & (mL < 5 * mL.median())).nonzero().squeeze(-1)
    keep = keep[torch.randperm(len(keep), generator=torch.Generator().manual_seed(0))][:CAP]
    XS = {a: v[keep] for a, v in XS.items()}
    XL = {k_: v[:, keep] for k_, v in XL.items()}
    groups = groups[keep]
    tr, te = linalg.split(len(groups), 0.2, 0, groups)
    tr_in, va_in = (tr[t] for t in linalg.split(len(tr), 0.2, 1, groups[tr]))
    sp = (tr_in, va_in, tr, te)
    entry = {"small": name, "n_tokens": len(groups), "layers": []}
    for s in s_layers:
        Y = XS[s]

        def pick(stack):
            vals = [(max(linalg.heldout_r2(X, Y, l_, tr_in, va_in) for l_ in GRID), i) for i, X in enumerate(stack)]
            return max(vals)[1]
        lt, lr = pick(XL["trained"]), pick(XL["random"])
        row = {"small_layer": s, "trained_layer": lt, "random_layer": lr,
               "trained_best": both(XL["trained"][lt], Y, *sp),
               "L_embedding": both(XL["trained"][0], Y, *sp),
               "random_best": both(XL["random"][lr], Y, *sp),
               "S_to_L_trained": both(Y, XL["trained"][lt], *sp)}
        for m in ("ridge", "rotation"):
            row[f"shared_part_{m}"] = round(row["trained_best"][m] - row["L_embedding"][m], 4)
        entry["layers"].append(row)
        log(name, json.dumps(row))
    res["pairs"].append(entry)
    OUT.write_text(json.dumps(res, indent=1))
    log(f"done {name} [{time.time() - t0:.0f}s]")
    del S, XS, XL
    gc.collect(); empty_cache(torch.device(DEV))
log("wrote", OUT)
