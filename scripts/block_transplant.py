"""Is the map F a map between *models* (weights), not just between representations?

Block transplant: replace S's decoder block s with L's block l, conjugated by linear maps:
    x_S --A--> L coordinates --L.block_l (L's own weights)--> --B--> S coordinates
and measure S's next-token loss on held-out texts. No training: forward passes + closed-form ridge.

S = Qwen2.5-0.5B-Instruct (24 blocks), L = Qwen2.5-1.5B-Instruct (28 blocks), same tokenizer, so every token is
aligned. l = matched relative depth. Conditions (block s output replaced for every position except position 0,
which keeps S's own output in every condition - it is the attention-sink token):
  intact            S unchanged
  skip              block removed (output = input): how much this block matters
  linear            best linear surrogate of S's own block (ridge x_in -> x_out): the bar L's block must beat
  transplant_pure   A (S_in -> L_in) and B (L_out -> S_out) fitted on natural activations only; L's block in between.
                    The strict test of "F conjugates weights".
  transplant_fit    same A and L block; B refit on the L block's outputs given mapped S inputs
  random_fit        as transplant_fit but with a random-init block of L's architecture (random features control)
Also: output fidelity (R^2 vs S's true block output) on held-out tokens.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/block_transplant.py
"""
import copy
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightmap import linalg
from weightmap.models import default_device, empty_cache

HERE = Path(__file__).parent
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
OUT = HERE / "block_transplant_results.json"
S_NAME, L_NAME = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
S_BLOCKS = [6, 12, 18]
K, MAXLEN = 64, 512
GRID = (1e-4, 1e-3, 1e-2, 1e-1)
DEV = default_device()
torch.manual_seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
texts = [t for t in texts if len(t.strip()) > 20]
tok = AutoTokenizer.from_pretrained(S_NAME)
ids = [tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids for t in texts]
tr_txt, te_txt = (x.tolist() for x in linalg.split(len(texts), 0.2, 0))
S = AutoModelForCausalLM.from_pretrained(S_NAME, dtype=torch.bfloat16).to(DEV).eval()
L = AutoModelForCausalLM.from_pretrained(L_NAME, dtype=torch.bfloat16).to(DEV).eval()
nS, nL = len(S.model.layers), len(L.model.layers)
PAIRS = {s: round(s / nS * nL) for s in S_BLOCKS}
log("texts train/test", len(tr_txt), len(te_txt), "block pairs S->L", PAIRS)


def random_block(l):
    b = copy.deepcopy(L.model.layers[l])
    std = L.config.initializer_range
    g = torch.Generator(device="cpu").manual_seed(l)
    for m in b.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data = (torch.randn(m.weight.shape, generator=g) * std).to(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif hasattr(m, "weight") and m.weight is not None and m.weight.dim() == 1:
            m.weight.data.fill_(1.0)
    return b


RAND = {s: random_block(l) for s, l in PAIRS.items()}


@torch.no_grad()
def states(model, x):
    return model.model(input_ids=x.to(DEV), output_hidden_states=True).hidden_states   # [0] = embedding


@torch.no_grad()
def run_block(block, h):
    """One L-architecture block on a full sequence h [1, T, d] (bs=1, no padding -> causal SDPA)."""
    h = h.to(torch.bfloat16)
    pid = torch.arange(h.shape[1], device=DEV)[None]
    pe = L.model.rotary_emb(h, pid)
    out = block(h, attention_mask=None, position_ids=pid, position_embeddings=pe)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------- sanity: calling L's block reproduces L
hL = states(L, ids[0])
l0 = PAIRS[S_BLOCKS[0]]
err = float((run_block(L.model.layers[l0], hL[l0]).float() - hL[l0 + 1].float()).norm() / hL[l0 + 1].float().norm())
log(f"sanity: L block {l0} re-run relative error {err:.4f}")

# ---------------------------------------------------------------- pass 1: natural activations
data = {s: {"s_in": [], "s_out": [], "l_in": [], "l_out": [], "grp": []} for s in S_BLOCKS}
pos_of = {}
for t in tr_txt:
    x = ids[t]
    T = x.shape[1]
    g = torch.Generator().manual_seed(t)
    pos = (torch.randperm(T - 1, generator=g)[:K] + 1).sort().values      # never position 0
    pos_of[t] = pos
    hS, hL = states(S, x), states(L, x)
    for s, l in PAIRS.items():
        d = data[s]
        d["s_in"].append(hS[s][0, pos].float().cpu()); d["s_out"].append(hS[s + 1][0, pos].float().cpu())
        d["l_in"].append(hL[l][0, pos].float().cpu()); d["l_out"].append(hL[l + 1][0, pos].float().cpu())
        d["grp"].append(torch.full((len(pos),), t))
    del hS, hL
empty_cache(torch.device(DEV))
for s in S_BLOCKS:
    data[s] = {k: torch.cat(v) for k, v in data[s].items()}


def fit(X, Y, grp):
    """Ridge with lambda chosen on an inner text-level split; refit on all rows. Inlier rows only."""
    nx, ny = X.norm(dim=1), Y.norm(dim=1)
    keep = ((nx < 5 * nx.median()) & (ny < 5 * ny.median())).nonzero().squeeze(-1)
    X, Y, grp = X[keep], Y[keep], grp[keep]
    tr, va = linalg.split(len(X), 0.2, 1, grp)
    best = max((linalg.heldout_r2(X, Y, lam, tr, va), lam) for lam in GRID)
    r = linalg.fit_ridge(X, Y, best[1])
    return {"W": r.W.float().to(DEV), "b": r.b.float().to(DEV), "val_r2": round(best[0], 4), "lam": best[1]}


def apply(m, h):
    return h.float() @ m["W"] + m["b"]


maps = {}
for s, l in PAIRS.items():
    d = data[s]
    maps[s] = {"A": fit(d["s_in"], d["l_in"], d["grp"]), "B_nat": fit(d["l_out"], d["s_out"], d["grp"]),
               "lin": fit(d["s_in"], d["s_out"], d["grp"])}
    log(f"S{s}<->L{l} val R2: A {maps[s]['A']['val_r2']}  B_nat {maps[s]['B_nat']['val_r2']}  "
        f"linear surrogate {maps[s]['lin']['val_r2']}")

# ---------------------------------------------------------------- pass 2: L block (and random block) on mapped inputs
z = {s: {"L": [], "R": []} for s in S_BLOCKS}
for t in tr_txt:
    hS = states(S, ids[t])
    for s in S_BLOCKS:
        hA = apply(maps[s]["A"], hS[s])
        z[s]["L"].append(run_block(L.model.layers[PAIRS[s]], hA)[0, pos_of[t]].float().cpu())
        z[s]["R"].append(run_block(RAND[s], hA)[0, pos_of[t]].float().cpu())
    del hS
empty_cache(torch.device(DEV))
for s in S_BLOCKS:
    maps[s]["B_fit"] = fit(torch.cat(z[s]["L"]), data[s]["s_out"], data[s]["grp"])
    maps[s]["B_rand"] = fit(torch.cat(z[s]["R"]), data[s]["s_out"], data[s]["grp"])
    log(f"S{s}: B_fit val R2 {maps[s]['B_fit']['val_r2']}  B_rand val R2 {maps[s]['B_rand']['val_r2']}")
del data, z


# ---------------------------------------------------------------- evaluation on held-out texts
def replacement(cond, s, h_in):
    if cond == "skip":
        return h_in.float()
    if cond == "linear":
        return apply(maps[s]["lin"], h_in)
    hA = apply(maps[s]["A"], h_in)
    if cond == "transplant_pure":
        return apply(maps[s]["B_nat"], run_block(L.model.layers[PAIRS[s]], hA))
    if cond == "transplant_fit":
        return apply(maps[s]["B_fit"], run_block(L.model.layers[PAIRS[s]], hA))
    if cond == "random_fit":
        return apply(maps[s]["B_rand"], run_block(RAND[s], hA))
    raise ValueError(cond)


CONDS = ["intact", "skip", "linear", "transplant_pure", "transplant_fit", "random_fit"]


@torch.no_grad()
def evaluate(s, cond):
    nll, n, fid_res, fid_tot = 0.0, 0, 0.0, 0.0
    captured = {}

    def hook(mod, args, kwargs, out):
        h_in = args[0] if args else kwargs["hidden_states"]
        true = out[0] if isinstance(out, tuple) else out
        new = replacement(cond, s, h_in).to(true.dtype)
        new[:, 0] = true[:, 0]                                            # attention-sink token keeps S's output
        captured["res"] = float(((new[0, 1:].float() - true[0, 1:].float()) ** 2).sum())
        captured["tot"] = float(((true[0, 1:].float() - true[0, 1:].float().mean(0)) ** 2).sum())
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    hd = None if cond == "intact" else S.model.layers[s].register_forward_hook(hook, with_kwargs=True)
    try:
        for t in te_txt:
            x = ids[t].to(DEV)
            lp = torch.log_softmax(S(input_ids=x).logits.float()[0, :-1], -1)
            nll -= float(lp.gather(-1, x[0, 1:, None]).sum()); n += x.shape[1] - 1
            if captured:
                fid_res += captured["res"]; fid_tot += captured["tot"]
    finally:
        if hd:
            hd.remove()
    empty_cache(torch.device(DEV))
    return {"nll": round(nll / n, 4), "fidelity_r2": round(1 - fid_res / fid_tot, 4) if fid_tot else 1.0}


res = {"setup": {"S": S_NAME, "L": L_NAME, "pairs": PAIRS, "n_train_texts": len(tr_txt),
                 "n_test_texts": len(te_txt), "k_per_text": K, "sanity_rel_err": round(err, 5)},
       "maps": {s: {k: {kk: vv for kk, vv in v.items() if kk in ("val_r2", "lam")} for k, v in maps[s].items()}
                for s in S_BLOCKS},
       "eval": {}}
for s in S_BLOCKS:
    r = {c: evaluate(s, c) for c in CONDS}
    base, skip = r["intact"]["nll"], r["skip"]["nll"]
    for c in CONDS:
        r[c]["recovered_of_skip_damage"] = round((skip - r[c]["nll"]) / (skip - base), 3) if skip != base else None
    res["eval"][s] = r
    log(f"S block {s} (L block {PAIRS[s]}):", json.dumps(r))
    OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
