"""Stage 3: representation map between L and S (closed-form, no training of either model).

1. Layer-by-layer linear CKA between S and L on identical token streams (shared tokenizer).
2. Ridge maps L->S and S->L at candidate layers: how much of S is linearly contained in L, and vice versa.
3. "Correct-vs-wrong" contrast directions from the *train* split target problems:
     d_S       = S-encoding(prompt + L's correct answer) - S-encoding(prompt + S's wrong answer)
     d_L->S    = Map_{L->S}( L-encoding(same pos) - L-encoding(same neg) )
   Same texts, different encoder. Tests whether L's representation, projected into S, gives a different/better edit.
"""
import numpy as np
from common import *

N_FIT_TOK = 15000

tokS, S = load(S_NAME)
tokL, L = load(L_NAME)
assert tokS.get_vocab() == tokL.get_vocab(), "tokenizers differ"
nS, nL = len(S.model.layers), len(L.model.layers)


def texts_for(task):
    data = split_data(task, "train")
    gS, gL = read(f"gen_S_{task}_train.json"), read(f"gen_L_{task}_train.json")
    return data, gS, gL


@torch.no_grad()
def encode(texts, model, layers, plen=None, bs=4, maxlen=1400, sample_k=None):
    """Returns per-text list of {layer: [T, d] cpu} for completion tokens (pos>=plen) or all pos>=1.
    sample_k: keep a random subset of k positions per text; the subset depends only on (text index, length),
    so S and L get identical positions (shared tokenizer)."""
    res = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tokS(chunk, return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(DEV)
        hs = hidden_states(model, enc)
        mask = enc.attention_mask.bool()
        for j in range(len(chunk)):
            valid = mask[j].nonzero().squeeze(-1)
            start = valid[0].item() + (plen[i + j] if plen else 1)
            idx = valid[valid >= start]
            if sample_k and len(idx) > sample_k:
                g = torch.Generator().manual_seed(i + j)
                idx = idx[torch.randperm(len(idx), generator=g)[:sample_k].sort().values.to(idx.device)]
            dt = torch.float16 if sample_k else torch.float32
            res.append({l: hs[l][j, idx].to(dt).cpu() for l in layers})
        del hs, enc
        torch.mps.empty_cache()
    return res


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))).item()


# ------------------------------------------------ fit texts: L's outputs on both train sets (generic in-domain text)
fit_texts = []
for task in ["math", "nq"]:
    data, gS, gL = texts_for(task)
    fit_texts += [prompt_text(tokS, task, d["q"]) + l["out"] for d, l in zip(data, gL)]
random.Random(0).shuffle(fit_texts)
log("fit texts", len(fit_texts))

encS = encode(fit_texts, S, list(range(nS - 1)), sample_k=40)
encL = encode(fit_texts, L, list(range(nL - 1)), sample_k=40)
XS = {l: torch.cat([e[l] for e in encS]) for l in range(nS - 1)}
XL = {l: torch.cat([e[l] for e in encL]) for l in range(nL - 1)}
del encS, encL
ntok = XS[0].shape[0]
log("tokens", ntok)

# drop "massive activation" tokens (attention sinks), judged on mid layer norms of both models
nmS = XS[nS // 2].float().norm(dim=1); nmL = XL[nL // 2].float().norm(dim=1)
keep = (nmS < 5 * nmS.median()) & (nmL < 5 * nmL.median())
g = torch.Generator().manual_seed(0)
idx = keep.nonzero().squeeze(-1)
idx = idx[torch.randperm(len(idx), generator=g)][:N_FIT_TOK]
for D in (XS, XL):
    for l in D:
        D[l] = D[l][idx].numpy()          # float16, converted to float64 lazily per layer
log("kept tokens", len(idx), "of", ntok)
f64 = lambda A: A.astype(np.float64)

# ------------------------------------------------ 1. CKA layer matrix
sub = slice(0, 5000)
C = np.array([[cka(f64(XS[a][sub]), f64(XL[b][sub])) for b in range(nL - 1)] for a in range(nS - 1)])
np.save(OUT / "cka.npy", C)
match = {a: int(C[a].argmax()) for a in range(nS - 1)}
log("CKA best L-layer per S-layer:", match)
log("CKA diag-ish max per S layer:", np.round(C.max(1), 3).tolist())


# ------------------------------------------------ 2. ridge maps
def ridge(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    W = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, my - mx @ W


def r2(X, Y, W, b):
    P = X @ W + b
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


ntr = int(0.8 * len(idx))
maps, map_stats = {}, {}
for a in S_LAYERS:
    b_ = match[a]
    stats = {}
    xs, xl, xs_prev = f64(XS[a]), f64(XL[b_]), f64(XS[a - 1])
    for direction, X, Y in [("L->S", xl, xs), ("S->L", xs, xl)]:
        best = None
        for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
            W, c = ridge(X[:ntr], Y[:ntr], lam * X.var(0).mean())
            s = r2(X[ntr:], Y[ntr:], W, c)
            if best is None or s > best[0]:
                best = (s, lam, W, c)
        stats[direction] = round(float(best[0]), 4)
        if direction == "L->S":
            maps[a] = (best[2], best[3])
    # control: shuffled pairing (map must use token correspondence)
    perm = np.random.default_rng(0).permutation(ntr)
    W, c = ridge(xl[:ntr][perm], xs[:ntr], 1e-2 * xl.var(0).mean())
    stats["L->S shuffled-control"] = round(float(r2(xl[ntr:], xs[ntr:], W, c)), 4)
    # control: S predicting itself from the previous layer (how good is "good")
    W, c = ridge(xs_prev[:ntr], xs[:ntr], 1e-3 * xs_prev.var(0).mean())
    stats["S(l-1)->S(l) reference"] = round(float(r2(xs_prev[ntr:], xs[ntr:], W, c)), 4)
    stats["L_layer"] = b_
    stats["cka"] = round(float(C[a, b_]), 4)
    map_stats[a] = stats
    log(f"S layer {a} <-> L layer {b_}:", stats)
save("map_stats.json", map_stats)
torch.save({a: (torch.tensor(W), torch.tensor(c)) for a, (W, c) in maps.items()}, OUT / "maps.pt")
resid_norm = {a: float(np.median(np.linalg.norm(f64(XS[a]), axis=1))) for a in S_LAYERS}
del XS, XL

# ------------------------------------------------ 3. contrast directions per task
dirs = {}
for task in ["math", "nq"]:
    data, gS, gL = texts_for(task)
    tgt = [(d, s, l) for d, s, l in zip(data, gS, gL) if l["ok"] and not s["ok"]]
    log(task, "train target pairs", len(tgt))
    prompts = [prompt_text(tokS, task, d["q"]) for d, _, _ in tgt]
    plen = [len(tokS(p).input_ids) for p in prompts]
    pos = [p + l["out"] for p, (_, _, l) in zip(prompts, tgt)]
    neg = [p + s["out"] for p, (_, s, _) in zip(prompts, tgt)]
    Lls = sorted({match[a] for a in S_LAYERS})

    def mean_rep(texts, model, layers):
        e = encode(texts, model, layers, plen=plen)
        # per-text mean over completion tokens, then mean over texts (each problem weighted equally)
        return {l: torch.stack([x[l].mean(0) for x in e if len(x[l])]).mean(0).double().numpy() for l in layers}

    pS, nS_ = mean_rep(pos, S, S_LAYERS), mean_rep(neg, S, S_LAYERS)
    pL, nL_ = mean_rep(pos, L, Lls), mean_rep(neg, L, Lls)
    dirs[task] = {}
    for a in S_LAYERS:
        W, _ = maps[a]
        dS = pS[a] - nS_[a]
        dLS = (pL[match[a]] - nL_[match[a]]) @ W
        cos = float(dS @ dLS / np.linalg.norm(dS) / np.linalg.norm(dLS))
        dirs[task][a] = {"S": torch.tensor(dS), "LtoS": torch.tensor(dLS), "cos": cos, "resid_norm": resid_norm[a]}
        log(f"  {task} layer {a}: cos(d_S, Map(d_L)) = {cos:.3f}   |d_S|={np.linalg.norm(dS):.2f} |Map d_L|={np.linalg.norm(dLS):.2f} resid={resid_norm[a]:.1f}")
torch.save(dirs, OUT / "dirs.pt")
save("layer_match.json", match)
log("done")
