"""Three quick, training-free follow-ups (all closed-form).

T1  Weight-level f: does a linear map send L's *weights* (word-embedding table) to S's?
    Qwen2.5-1.5B -> 0.5B (same vocab) and Qwen2.5-1.5B -> SmolLM2-360M (shared token strings).
T2  Linear or non-linear f? Qwen 0.5B <- 1.5B at S10<-L13 / S14<-L17, text-level holdout:
    linear ridge vs linear + random Fourier features (closed-form kernel approximation).
T3  Is S like a *pruned* L? Predict S from a subset of 896 of L's 1536 dims chosen by axis
    (pruning-like) vs the top-896 principal components of L (rotated subspace of the same size)
    vs 896 random dims vs all 1536.

Run:  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
      uv run python scripts/quick_tests.py
"""
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightmap import WeightMap, linalg
from weightmap.core import _inlier_mask

HERE = Path(__file__).parent
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
OUT = HERE / "quick_tests_results.json"
QL, QS, SM = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M"
GRID = (1e-4, 1e-3, 1e-2, 1e-1)
res = {}
torch.manual_seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ridge_fast(Xtr, Ytr, Xte, Yte, return_best_lam=False):
    """Closed-form ridge. Heavy products in float32 (fast on Apple AMX); the small d x d solve in float64.
    Lambda grid relative to mean feature variance x n, same convention as weightmap.linalg.fit_ridge."""
    Xtr, Ytr, Xte, Yte = (t.float() for t in (Xtr, Ytr, Xte, Yte))
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    G, C = (Xc.T @ Xc).double(), (Xc.T @ Yc).double()
    scale = float(Xtr.var(0).mean()) * len(Xtr)
    tot = float(((Yte - Yte.mean(0)) ** 2).sum())
    best, best_lam = -1e9, None
    for lam in GRID:
        W = torch.linalg.solve(G + lam * scale * torch.eye(G.shape[0], dtype=torch.float64), C).float()
        r2 = 1 - float(((Yte - ((Xte - mx) @ W + my)) ** 2).sum()) / tot
        if r2 > best:
            best, best_lam = r2, lam
    return (round(best, 4), best_lam) if return_best_lam else round(best, 4)


def ridge_report(X, Y, tr, te):
    r2, lam = ridge_fast(X[tr], Y[tr], X[te], Y[te], return_best_lam=True)
    perm = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(1))]
    global GRID
    keep, GRID = GRID, (lam,)
    shuf = ridge_fast(X[perm], Y[tr], X[te], Y[te])        # pairing broken in training rows
    GRID = keep
    return r2, shuf


# ============================================================ T1: embedding tables
def emb(name):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    t = AutoTokenizer.from_pretrained(name)
    E = m.get_input_embeddings().weight.detach().float()
    tied = m.get_output_embeddings().weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    del m
    return E, t, tied


log("T1 embeddings")
EL, tokL, tiedL = emb(QL)
ES, tokS, tiedS = emb(QS)
n = len(tokS)                                       # real token ids (rows past this are padding)
ids = torch.randperm(n)[:40000]
tr, te = linalg.split(len(ids), 0.2, 0)
XL, XS = EL[ids], ES[ids]
r_ls, sh_ls = ridge_report(XL, XS, tr, te)
r_sl, sh_sl = ridge_report(XS, XL, tr, te)
res["T1_qwen1.5b_to_qwen0.5b_embeddings"] = {"n_tokens": len(ids), "tied_embeddings": [tiedL, tiedS],
                                             "r2_L_to_S": r_ls, "r2_S_to_L": r_sl, "r2_shuffled": sh_ls}
log(res["T1_qwen1.5b_to_qwen0.5b_embeddings"])

ESm, tokSm, tiedSm = emb(SM)
vL, vSm = tokL.get_vocab(), tokSm.get_vocab()
shared = sorted(set(vL) & set(vSm))
iL = torch.tensor([vL[s] for s in shared]); iSm = torch.tensor([vSm[s] for s in shared])
tr, te = linalg.split(len(shared), 0.2, 0)
XL, XSm = EL[iL], ESm[iSm]
r_ls, sh_ls = ridge_report(XL, XSm, tr, te)
r_sl, _ = ridge_report(XSm, XL, tr, te)
res["T1_qwen1.5b_to_smollm2-360m_embeddings"] = {"n_shared_token_strings": len(shared), "tied_embeddings": tiedSm,
                                                 "r2_L_to_S": r_ls, "r2_S_to_L": r_sl, "r2_shuffled": sh_ls}
log(res["T1_qwen1.5b_to_smollm2-360m_embeddings"])
del EL, ES, ESm, XL, XS, XSm

# ============================================================ activations for T2 / T3
texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
wm = WeightMap(large=QL, small=QS)
PAIRS = [(10, 13), (14, 17)]
log("collecting activations")
A_S, A_L, groups = wm._collect(texts, [p[0] for p in PAIRS], [p[1] for p in PAIRS], 40, 4, 768, 0)
del wm


def rff(X, W, b):
    return torch.cos(X @ W + b) * math.sqrt(2.0 / W.shape[1])


def ridge_gram_cv(Ztr, Ytr, Zte, Yte):
    return ridge_fast(Ztr, Ytr, Zte, Yte)


for s, l in PAIRS:
    xs, xl = A_S[s].float(), A_L[l].float()
    keep = _inlier_mask(xs, xl, 5.0)
    idx = keep.nonzero().squeeze(-1)
    idx = idx[torch.randperm(len(idx))][:15000]
    xs, xl, g = xs[idx], xl[idx], groups[idx]
    tr, te = linalg.split(len(idx), 0.2, 0, g)
    key = f"S{s}<-L{l}"
    # ---------------- T2: linear vs non-linear
    mu, sd = xl[tr].mean(0), xl[tr].std(0) + 1e-6
    zl = (xl - mu) / sd
    lin = ridge_gram_cv(zl[tr], xs[tr], zl[te], xs[te])
    sub = zl[tr][torch.randperm(len(tr))[:2000]]
    med = float(torch.cdist(sub, sub).median())
    t2 = {"linear": lin}
    for D in (4096,):
        for mult in (0.5, 1.0, 2.0):
            W = torch.randn(zl.shape[1], D) / (mult * med)
            b = torch.rand(D) * 2 * math.pi
            Z = torch.cat([zl, rff(zl, W, b)], 1)            # linear + non-linear features (superset of linear)
            t2[f"linear+rff D={D} bw={mult}xmedian"] = ridge_gram_cv(Z[tr], xs[tr], Z[te], xs[te])
            del Z
    t2["best_nonlinear_gain"] = round(max(v for k, v in t2.items() if k.startswith("linear+rff")) - lin, 4)
    log("T2", key, t2)
    # ---------------- T3: axis selection (pruning-like) vs rotated subspace of the same size
    k = xs.shape[1]                                         # 896 = S's width
    zc, yc = zl[tr] - zl[tr].mean(0), xs[tr] - xs[tr].mean(0)
    Gf = (zc.T @ zc).double()
    Wf = torch.linalg.solve(Gf + 1e-3 * float(zl[tr].var(0).mean()) * len(tr) * torch.eye(Gf.shape[0], dtype=torch.float64), (zc.T @ yc).double())
    row_imp = Wf.norm(dim=1)                            # importance of each L dim in the full map
    sel = row_imp.topk(k).indices
    rnd = torch.randperm(zl.shape[1])[:k]
    U, S_, Vh = torch.linalg.svd(zl[tr] - zl[tr].mean(0), full_matrices=False)
    P = Vh[:k].T                                            # top-k principal directions of L
    t3 = {"all_1536_dims": lin,
          f"pca_top{k}_rotated": ridge_gram_cv(zl[tr] @ P, xs[tr], zl[te] @ P, xs[te]),
          f"select_{k}_axes_by_importance": ridge_gram_cv(zl[tr][:, sel], xs[tr], zl[te][:, sel], xs[te]),
          f"random_{k}_axes": ridge_gram_cv(zl[tr][:, rnd], xs[tr], zl[te][:, rnd], xs[te])}
    for kk in (448, 224):
        sel_k = row_imp.topk(kk).indices
        t3[f"select_{kk}_axes_by_importance"] = ridge_gram_cv(zl[tr][:, sel_k], xs[tr], zl[te][:, sel_k], xs[te])
        t3[f"pca_top{kk}_rotated"] = ridge_gram_cv(zl[tr] @ Vh[:kk].T, xs[tr], zl[te] @ Vh[:kk].T, xs[te])
    log("T3", key, t3)
    res[f"T2_{key}"] = t2
    res[f"T3_{key}"] = t3
    res[f"n_tokens_{key}"] = {"train": len(tr), "test": len(te)}

OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
