"""Port a weight-space program (Barry-style sparse LP edit) from one model to another through the map.

Barry's edit on a down_proj:  dW = delta * u v^T, u = loop direction (residual space), v = sparse trigger over MLP
columns found by an LP on base activations h_t (MLP hidden):
    min ||v||_1 + lam * sum(slack)   s.t.  v.h_t <= -1 on loop tokens R,  |v.h_t| <= eps on productive tokens P
then keep 8 columns (largest |v_j| std(h_j)) and re-solve on those only.

S = SmolLM2-135M (loops on GSM8K), L = Qwen2.5-1.5B-Instruct, edit at S block 18 (L block matched by the map).
Edits on S, all with delta = 2|d_S| (the steering strength that worked in loop_diagnostic):
  S-native   u = d_S (S's own loop direction),  v = LP on S's MLP columns with S's labels
  ported     u = Map(d_L) (L's direction carried by the activation map),
             v = L's own LP edit (on L's MLP columns, 8 columns) carried into S's columns by a *label-free* LP
                 that makes 8 S columns reproduce L's trigger values on the same text (neuron-level port)
  random     u = d_S, 8 random S columns, random signs, scaled to S-native's mean |trigger| on loop tokens
Metrics: held-out GSM8K loop rate / accuracy / early coherence (as loop_diagnostic) + trigger AUC on held-out tokens.
No training: forward passes, closed-form ridge, linear programs.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/lp_port.py
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack

from weightmap import WeightMap, align
from weightmap.models import empty_cache, model_device

HERE = Path(__file__).parent
# SWEEP=1: tune the edit strength on a separate GSM8K tune slice (S-native edit), then evaluate every edit at that
# strength on the test slice, adding the hybrid edit (S's own trigger columns + L's mapped direction).
SWEEP = os.environ.get("SWEEP") == "1"
SCALES = [2.0, 4.0, 8.0, 16.0]
OUT = HERE / ("lp_port_sweep_results.json" if SWEEP else "lp_port_results.json")
S_NAME, L_NAME = "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-1.5B-Instruct"
S_LAYER, MAX_NEW, W, SCALE = 18, 256, 16, 2.0
N_DIR, N_TEST, N_COLS, EPS, LAM = 400, 200, 8, 0.1, 1.0
N_TOK, PREFILTER = 400, 1000                   # tokens per class for the LP; columns pre-screened (both models) for LP size
torch.manual_seed(0)
rng = np.random.default_rng(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------ GSM8K + loop detector (identical to loop_diagnostic.py)
def prompt(q):
    return f"Question: {q}\nAnswer: Let's think step by step."


def gold(a):
    return a.split("####")[-1].strip().replace(",", "")


def answer(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", t.split("\nQuestion:")[0])
    return m[-1].replace(",", "").rstrip(".") if m else ""


def loop_start(ids):
    seen, rep = set(), []
    for t in range(len(ids)):
        if t + 1 < W:
            rep.append(False); continue
        w = tuple(ids[t + 1 - W:t + 1]); rep.append(w in seen); seen.add(w)
    n, suf, start = len(rep), 0, None
    for t in range(n - 1, -1, -1):
        suf += rep[t]
        if n - t >= 64 and suf / (n - t) >= 0.8:
            start = t
    return start


train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(N_DIR))
test_all = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0)
tune, test = test_all.select(range(60)), test_all.select(range(60, 60 + N_TEST))
wm = WeightMap(large=L_NAME, small=S_NAME)
S, L, tokS, tokL = wm.small, wm.large, wm.tok_small, wm.tok_large
devS, devL = model_device(S), model_device(L)


def run(ds):
    ps = [prompt(r["question"]) for r in ds]
    outs = wm.generate(ps, max_new_tokens=MAX_NEW, batch_size=40)
    loops = [loop_start(tokS(o, add_special_tokens=False).input_ids) is not None for o in outs]
    acc = [answer(o) == gold(r["answer"]) for o, r in zip(outs, ds)]
    return ps, outs, loops, acc


@torch.no_grad()
def early_nll(ps, outs, n_tok=48):
    tot, cnt = 0.0, 0
    for p, o in zip(ps, outs):
        comp = tokL(o, add_special_tokens=False).input_ids[:n_tok]
        if not comp:
            continue
        pid = tokL(p).input_ids
        lp = torch.log_softmax(L(torch.tensor([pid + comp], device=devL)).logits.float()[0, len(pid) - 1:-1], -1)
        tot -= float(lp.gather(-1, torch.tensor(comp, device=devL).unsqueeze(-1)).sum()); cnt += len(comp)
    empty_cache(devL)
    return tot / max(cnt, 1)


def summarize(name, ps, outs, loops, acc):
    r = {"loop_rate": round(sum(loops) / len(loops), 4), "accuracy": round(sum(acc) / len(acc), 4),
         "early_nll_qwen": round(early_nll(ps, outs), 4)}
    log(name, r)
    return r


# ------------------------------------------------ 1. label S's own outputs; fit the map; loop directions
ps, outs, loops, _ = run(train)
items = []                                       # (text, region_start_char, is_loop)
for p, o in zip(ps, outs):
    ids = tokS(o, add_special_tokens=False).input_ids
    st = loop_start(ids)
    if st is None:
        items.append((p + o, len(p), False))
    else:
        items.append((p + o, len(p) + len(tokS.decode(ids[:st], skip_special_tokens=True)), True))
loop_items = [x for x in items if x[2]]
prod_items = [x for x in items if not x[2]]
log("loop texts", len(loop_items), "productive", len(prod_items))
wm.fit([p + o for p, o in zip(ps, outs)], layers=[S_LAYER], positions_per_text=32, batch_size=4, max_length=512,
       cka_texts=100, max_tokens=12000)
L_LAYER = wm.layer_match[S_LAYER]
log("map", wm.report())
dS = (wm.encode([t for t, _, _ in loop_items], S_LAYER, "small", [c for _, c, _ in loop_items]).mean(0)
      - wm.encode([t for t, _, _ in prod_items], S_LAYER, "small", [c for _, c, _ in prod_items]).mean(0))
dL = (wm.encode([t for t, _, _ in loop_items], L_LAYER, "large", [c for _, c, _ in loop_items]).mean(0)
      - wm.encode([t for t, _, _ in prod_items], L_LAYER, "large", [c for _, c, _ in prod_items]).mean(0))
dLS = wm.transport(dL, S_LAYER)
cos_dir = float(torch.nn.functional.cosine_similarity(dS, dLS, dim=0))
log(f"S{S_LAYER}/L{L_LAYER}: cos(d_S, Map(d_L)) = {cos_dir:.3f}")


# ------------------------------------------------ 2. MLP-hidden activations of both models at aligned tokens
@torch.no_grad()
def mlp_hidden(model, layer, enc, dev):
    store = {}
    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0]))
    try:
        model.model(**{k: v.to(dev) for k, v in enc.items()})
    finally:
        h.remove()
    return store["h"]


@torch.no_grad()
def collect(group, per_text):
    """Aligned tokens after each text's region start: (S MLP hidden, L MLP hidden, text id)."""
    HS, HL, G = [], [], []
    for i in range(0, len(group), 4):
        chunk = group[i:i + 4]
        txt = [t for t, _, _ in chunk]
        enc_s, off_s = align.tokenize(tokS, txt, 512)
        enc_l, off_l = align.tokenize(tokL, txt, 512)
        hs, hl = mlp_hidden(S, S_LAYER, enc_s, devS), mlp_hidden(L, L_LAYER, enc_l, devL)
        for j, (_, c0, _) in enumerate(chunk):
            pairs = [(a, b) for a, b in align.boundary_pairs(off_s[j], enc_s["attention_mask"][j].bool(),
                                                             off_l[j], enc_l["attention_mask"][j].bool())
                     if int(off_s[j][a, 1]) > c0]
            if not pairs:
                continue
            sel = rng.choice(len(pairs), min(per_text, len(pairs)), replace=False)
            a = torch.tensor([pairs[k][0] for k in sel], device=devS)
            b = torch.tensor([pairs[k][1] for k in sel], device=devL)
            HS.append(hs[j, a].float().cpu()); HL.append(hl[j, b].float().cpu()); G += [i + j] * len(sel)
        del hs, hl
        empty_cache(devS)
    return torch.cat(HS).numpy(), torch.cat(HL).numpy(), np.array(G)


R_S, R_L, R_G = collect(loop_items, 12)
P_S, P_L, P_G = collect(prod_items, 4)
log("tokens: loop", len(R_S), "productive", len(P_S))


def split_by_text(G, frac=0.2, seed=0):
    u = np.unique(G)
    te = set(np.random.default_rng(seed).choice(u, max(1, int(frac * len(u))), replace=False).tolist())
    m = np.array([g in te for g in G])
    return ~m, m


rtr, rte = split_by_text(R_G)
ptr, pte = split_by_text(P_G, seed=1)


def sub(X, mask, n):
    idx = np.flatnonzero(mask)
    return X[rng.choice(idx, min(n, len(idx)), replace=False)] if len(idx) > n else X[idx]


# ------------------------------------------------ 3. the LPs
def solve_lp(A_le, b_le, n_var, n_slack, slack_rows):
    """min sum|v| + LAM*sum(slack); rows: A_le @ [v+, v-] - slack <= b_le."""
    c = np.concatenate([np.ones(2 * n_var), LAM * np.ones(n_slack)])
    S_mat = csr_matrix((-np.ones(len(slack_rows)), (np.arange(len(slack_rows)), slack_rows)),
                       shape=(len(slack_rows), n_slack))
    A = hstack([csr_matrix(np.hstack([A_le, -A_le])), S_mat]).tocsr()
    r = linprog(c, A_ub=A, b_ub=b_le, bounds=(0, None), method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    return r.x[:n_var] - r.x[n_var:2 * n_var]


def loop_lp(HR, HP, cols):
    """Barry's LP restricted to `cols`: v.h <= -1 on R, |v.h| <= EPS on P (each row has its own slack)."""
    R, P = HR[:, cols], HP[:, cols]
    nR, nP = len(R), len(P)
    A = np.vstack([R, P, -P]); b = np.concatenate([-np.ones(nR), EPS * np.ones(2 * nP)])
    slack_rows = np.concatenate([np.arange(nR), nR + np.arange(nP), nR + np.arange(nP)])
    v = solve_lp(A, b, len(cols), nR + nP, slack_rows)
    return v


def match_lp(HS_, y, cols):
    """Label-free port: |v.h_S - y| <= EPS on every token (y = L's trigger value at the aligned token)."""
    X = HS_[:, cols]
    n = len(X)
    A = np.vstack([X, -X]); b = np.concatenate([y + EPS, -y + EPS])
    return solve_lp(A, b, len(cols), n, np.concatenate([np.arange(n), np.arange(n)]))


def sparse_edit(fn, H_all, *args, cols=None):
    """Full LP -> keep N_COLS columns by |v_j| std(h_j) -> re-solve on those only. Returns (cols, v)."""
    cols = np.arange(H_all.shape[1]) if cols is None else cols
    t0 = time.time()
    v = fn(*args, cols)
    score = np.abs(v) * H_all[:, cols].std(0)
    keep = cols[np.argsort(-score)[:N_COLS]]
    v8 = fn(*args, keep)
    log(f"  LP over {len(cols)} cols: {int((np.abs(v) > 1e-6).sum())} nonzero -> {N_COLS} cols [{time.time() - t0:.0f}s]")
    return keep, v8


def auc(pos, neg):
    """P(score on a loop token < score on a productive token): trigger pushes negative on loops."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[len(pos):].sum() - len(neg) * (len(neg) + 1) / 2) / (len(pos) * len(neg)))


HR_S, HP_S = sub(R_S, rtr, N_TOK), sub(P_S, ptr, N_TOK)


def prescreen(R, P):
    eff = np.abs(R[rtr].mean(0) - P[ptr].mean(0)) / (np.vstack([R[rtr], P[ptr]]).std(0) + 1e-6)
    return np.argsort(-eff)[:PREFILTER]


log("S-native LP")
cols_S, v_S = sparse_edit(loop_lp, np.vstack([HR_S, HP_S]), HR_S, HP_S, cols=prescreen(R_S, P_S))

# L's own LP on its MLP columns (pre-screened by effect size so HiGHS stays fast), same tokens
HR_L, HP_L = sub(R_L, rtr, N_TOK), sub(P_L, ptr, N_TOK)
log("L LP")
cols_L, v_L = sparse_edit(loop_lp, np.vstack([HR_L, HP_L]), HR_L, HP_L, cols=prescreen(R_L, P_L))

# label-free neuron-level port of L's trigger into S's columns (all training tokens, labels unused)
Xtr_S = np.vstack([R_S[rtr], P_S[ptr]]); Xtr_L = np.vstack([R_L[rtr], P_L[ptr]])
pick = rng.choice(len(Xtr_S), min(2 * N_TOK, len(Xtr_S)), replace=False)
y = Xtr_L[pick][:, cols_L] @ v_L
log("port LP")
cols_P, v_P = sparse_edit(match_lp, Xtr_S[pick], Xtr_S[pick], y)

cols_R = rng.choice(S.config.intermediate_size, N_COLS, replace=False)
v_R = rng.choice([-1.0, 1.0], N_COLS)
v_R *= np.abs(R_S[rtr][:, cols_S] @ v_S).mean() / np.abs(R_S[rtr][:, cols_R] @ v_R).mean()


def trig(H, cols, v):
    return H[:, cols] @ v


res = {"setup": {"S": S_NAME, "L": L_NAME, "S_layer": S_LAYER, "L_layer": L_LAYER, "n_cols": N_COLS, "eps": EPS,
                 "lam": LAM, "scale_x_dS_norm": SCALE, "n_test": N_TEST, "prefilter": PREFILTER},
       "direction_cosine": round(cos_dir, 4), "map": wm.report(),
       "columns": {"S_native": cols_S.tolist(), "L": cols_L.tolist(), "ported": cols_P.tolist(),
                   "random": cols_R.tolist(), "overlap_S_native_vs_ported": len(set(cols_S) & set(cols_P))},
       "trigger_auc_heldout": {
           "S_native": round(auc(trig(R_S[rte], cols_S, v_S), trig(P_S[pte], cols_S, v_S)), 4),
           "L_in_L": round(auc(trig(R_L[rte], cols_L, v_L), trig(P_L[pte], cols_L, v_L)), 4),
           "ported": round(auc(trig(R_S[rte], cols_P, v_P), trig(P_S[pte], cols_P, v_P)), 4),
           "random": round(auc(trig(R_S[rte], cols_R, v_R), trig(P_S[pte], cols_R, v_R)), 4)},
       "port_fidelity_r2": None}
yt = np.vstack([R_L[rte], P_L[pte]])[:, cols_L] @ v_L
yp = np.vstack([R_S[rte], P_S[pte]])[:, cols_P] @ v_P
res["port_fidelity_r2"] = round(float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 4)
log("trigger AUC (held-out tokens)", res["trigger_auc_heldout"], "port fidelity R2", res["port_fidelity_r2"])

# ------------------------------------------------ 3b. DENSE=1: is the port limited by using only 8 neurons?
# The port above keeps 8 S neurons chosen by an L1 program. Here L's trigger is fitted from ALL of S's MLP neurons
# with ridge (lambda on an inner split by text), and from the top-k neurons of that dense fit. If the dense fit is
# also poor, L's trigger is not linearly readable from S's neurons at all; if it is good, the 8-neuron limit is
# what fails. Writes lp_port_dense_results.json and exits before any generation.
if os.environ.get("DENSE") == "1":
    def r2_of(y_, p_):
        return float(1 - ((y_ - p_) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum())

    Xtr = np.vstack([R_S[rtr], P_S[ptr]]).astype(np.float64)
    ytr = np.vstack([R_L[rtr], P_L[ptr]])[:, cols_L].astype(np.float64) @ v_L
    Xte = np.vstack([R_S[rte], P_S[pte]]).astype(np.float64)
    gtr = np.concatenate([R_G[rtr], P_G[ptr] + 10 ** 6])                # text id of each training token
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ug = np.unique(gtr)
    va_texts = set(np.random.default_rng(0).choice(ug, max(1, len(ug) // 5), replace=False).tolist())
    va = np.array([g in va_texts for g in gtr])

    def ridge(Zf, yf, lam_):
        ym = yf.mean()
        w_ = np.linalg.solve(Zf.T @ Zf + lam_ * len(Zf) * np.eye(Zf.shape[1]), Zf.T @ (yf - ym))
        return w_, ym

    def fit_eval(cols_):
        best = max((r2_of(ytr[va], Z[va][:, cols_] @ ridge(Z[~va][:, cols_], ytr[~va], lam_)[0]
                          + ytr[~va].mean()), lam_) for lam_ in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0))
        w_, ym = ridge(Z[:, cols_], ytr, best[1])
        pred = Zte[:, cols_] @ w_ + ym
        nR = int(rte.sum())
        return {"r2_heldout": round(r2_of(yt, pred), 4), "lambda": best[1],
                "auc_loop_vs_productive": round(auc(pred[:nR], pred[nR:]), 4)}, w_

    dense, w_all = fit_eval(np.arange(Z.shape[1]))
    order = np.argsort(-np.abs(w_all))
    out = {"n_train_tokens": int(len(Z)), "n_test_tokens": int(len(Zte)), "n_S_neurons": int(Z.shape[1]),
           "lp_8_neurons": {"r2_heldout": res["port_fidelity_r2"], "auc_loop_vs_productive": res["trigger_auc_heldout"]["ported"]},
           "ridge_on_the_8_lp_neurons": fit_eval(np.asarray(cols_P))[0],
           "ridge_all_neurons": dense,
           "ridge_top_k_neurons_of_dense_fit": {str(k_): fit_eval(order[:k_])[0] for k_ in (8, 32, 128, 512)},
           "reference_auc": {"L_trigger_in_L": res["trigger_auc_heldout"]["L_in_L"],
                             "S_own_trigger": res["trigger_auc_heldout"]["S_native"]}}
    (HERE / "lp_port_dense_results.json").write_text(json.dumps(out, indent=1))
    log("DENSE", json.dumps(out))
    raise SystemExit

# ------------------------------------------------ 4. apply each edit to S's weights, generate held-out
down = S.model.layers[S_LAYER].mlp.down_proj
W0 = down.weight.data.clone()


def apply_edit(u, cols, v, scale):
    u = (u / u.norm()).float().to(devS)
    Wf = W0.float().clone()
    Wf[:, torch.as_tensor(cols, device=devS)] += (scale * float(dS.norm())) * u[:, None] \
        * torch.as_tensor(v, dtype=torch.float32, device=devS)[None]
    down.weight.data = Wf.to(W0.dtype)


EDITS = [("S_native", dS, cols_S, v_S), ("hybrid_S_trigger_L_direction", dLS, cols_S, v_S),
         ("ported", dLS, cols_P, v_P), ("random", dS, cols_R, v_R)]
# ------------------------------------------------ 4b. FOLLOWUP=1: does L's trigger work as an edit when it may
# use every S neuron? The 8-neuron port fails (above), but a dense readout reproduces L's trigger (DENSE=1).
# Here the port program runs without the 8-neuron limit and the result is applied as a rank-1 edit. The payload
# is S's own loop direction in every condition, so only the trigger differs. Writes lp_port_followup_results.json.
if os.environ.get("FOLLOWUP") in ("1", "2", "3"):
    allc = np.arange(Xtr_S.shape[1])
    v_full = match_lp(Xtr_S[pick], y, allc)                       # L's trigger over all of S's neurons, no labels
    colsS_full = prescreen(R_S, P_S)
    v_Sfull = loop_lp(HR_S, HP_S, colsS_full)                     # S's own trigger before the reduction to 8
    Xte_S = np.vstack([R_S[rte], P_S[pte]])
    nR = int(rte.sum())

    def fid(pred):
        return {"r2_vs_L_trigger": round(float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 4),
                "auc_loop_vs_productive": round(auc(pred[:nR], pred[nR:]), 4)}

    fu = {"ported_trigger_all_neurons": {"nonzero_weights": int((np.abs(v_full) > 1e-6).sum()), **fid(Xte_S @ v_full)},
          "S_own_trigger_unreduced": {"nonzero_weights": int((np.abs(v_Sfull) > 1e-6).sum()),
                                      "auc_loop_vs_productive": round(auc((Xte_S[:, colsS_full] @ v_Sfull)[:nR],
                                                                          (Xte_S[:, colsS_full] @ v_Sfull)[nR:]), 4)},
          "payload": "S's own loop direction in every condition"}

    # FOLLOWUP=2: only the trigger statistics. A dense trigger that fires on every token would act as constant
    # steering, not as a gate, so report how each trigger fires on held-out looping and productive tokens.
    def fire(cols, v):
        tv = Xte_S[:, cols] @ np.asarray(v)
        return {"mean_on_looping": round(float(tv[:nR].mean()), 3), "mean_on_productive": round(float(tv[nR:].mean()), 3),
                "sd_on_productive": round(float(tv[nR:].std()), 3), "auc_loop_vs_productive": round(auc(tv[:nR], tv[nR:]), 4)}
    fu["trigger_firing_heldout_tokens"] = {
        "S_own_8_neurons": fire(cols_S, v_S), "S_own_unreduced": fire(colsS_full, v_Sfull),
        "ported_8_neurons": fire(cols_P, v_P), "ported_all_neurons": fire(allc, v_full),
        "L_trigger_in_L": {"mean_on_looping": round(float(yt[:nR].mean()), 3), "mean_on_productive": round(float(yt[nR:].mean()), 3),
                           "sd_on_productive": round(float(yt[nR:].std()), 3)}}
    if os.environ.get("FOLLOWUP") == "2":
        (HERE / "lp_port_trigger_firing.json").write_text(json.dumps(fu["trigger_firing_heldout_tokens"], indent=1))
        log("FIRING", json.dumps(fu["trigger_firing_heldout_tokens"]))
        raise SystemExit

    # FOLLOWUP=3: is the dense ported edit's effect due to L's trigger, or would any dense trigger of the same
    # size do? Three unselective controls over the same payload (S's own direction) and strength, each scaled to
    # fire as strongly as the real ported trigger on looping training tokens. Writes lp_port_dense_controls.json.
    if os.environ.get("FOLLOWUP") == "3":
        Rtr = R_S[rtr]
        target = float(np.abs(Rtr @ v_full).mean())

        def matched(v):
            m_ = float(np.abs(Rtr @ v).mean())
            return v * (target / m_) if m_ > 0 else v

        nz = np.flatnonzero(np.abs(v_full) > 1e-6)
        v_shuf_w = np.zeros_like(v_full)
        v_shuf_w[nz] = np.random.default_rng(2).permutation(v_full[nz])            # real weights, shuffled over the same neurons
        ctrl = {"ported_all_neurons (real)": v_full,
                "port_program_on_shuffled_targets": matched(match_lp(Xtr_S[pick], y[np.random.default_rng(1).permutation(len(y))], allc)),
                "real_weights_shuffled_over_same_neurons": matched(v_shuf_w),
                "real_weights_random_signs": matched(v_full * np.random.default_rng(3).choice([-1.0, 1.0], len(v_full)))}
        dc = {"payload": "S's own loop direction", "strength": 2.0,
              "firing": {k_: {"nonzero": int((np.abs(v_) > 1e-6).sum()), **fire(allc, v_)} for k_, v_ in ctrl.items()},
              "test": {"base": summarize("TEST base", *run(test))}}
        for k_, v_ in ctrl.items():
            apply_edit(dS, allc, v_, 2.0)
            dc["test"][k_] = summarize(f"TEST {k_} x2", *run(test))
            down.weight.data = W0.clone()
            (HERE / "lp_port_dense_controls.json").write_text(json.dumps(dc, indent=1, default=float))
        log("FOLLOWUP 3 wrote lp_port_dense_controls.json")
        raise SystemExit
    fu["test"] = {"base": summarize("TEST base", *run(test))}
    for name, cols, v in (("S_own_trigger_8_neurons", cols_S, v_S), ("S_own_trigger_unreduced", colsS_full, v_Sfull),
                          ("ported_trigger_8_neurons", cols_P, v_P), ("ported_trigger_all_neurons", allc, v_full)):
        for sc in (2.0, 4.0):
            apply_edit(dS, cols, v, sc)
            fu["test"][f"{name} x{sc}"] = summarize(f"TEST {name} x{sc}", *run(test))
            down.weight.data = W0.clone()
            fu["test"][f"{name} x{sc}"]["weights_changed"] = int((np.abs(np.asarray(v)) > 1e-6).sum() * W0.shape[0])
        (HERE / "lp_port_followup_results.json").write_text(json.dumps(fu, indent=1, default=float))
    log("FOLLOWUP wrote lp_port_followup_results.json")
    raise SystemExit

best_scale = SCALE
if SWEEP:
    res["tune"] = {"base": summarize("TUNE base", *run(tune))}
    cands = []
    for sc in SCALES:
        apply_edit(dS, cols_S, v_S, sc)
        r = summarize(f"TUNE S_native x{sc}", *run(tune))
        down.weight.data = W0.clone()
        res["tune"][f"S_native x{sc}"] = r
        # same rule as loop_diagnostic: lowest loop rate with early coherence within +15% of base
        if r["early_nll_qwen"] <= res["tune"]["base"]["early_nll_qwen"] * 1.15:
            cands.append((r["loop_rate"], -r["accuracy"], sc))
    best_scale = min(cands)[2] if cands else SCALES[0]
    log("chosen strength", best_scale)
else:
    EDITS = [e for e in EDITS if not e[0].startswith("hybrid")]
res["setup"]["chosen_scale"] = best_scale
res["test"] = {"base": summarize("TEST base", *run(test))}
for name, u, cols, v in EDITS:
    apply_edit(u, cols, v, best_scale)
    res["test"][name] = summarize(f"TEST {name} x{best_scale}", *run(test))
    down.weight.data = W0.clone()
    res["test"][name]["weights_changed"] = int(len(cols) * W0.shape[0])
OUT.write_text(json.dumps(res, indent=1, default=float))
log("wrote", OUT)
