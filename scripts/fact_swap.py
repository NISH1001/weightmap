"""Fact-swap test with token-identity controls.

A donor model's hidden state at the last subject token of fact i is mapped into a recipient model and written
over the recipient's state at the last subject token of a different fact j (same relation, different answer).
Success = the recipient's next token is fact i's answer. Facts: CounterFact (NeelNanda/counterfact-tracing).

Maps (closed form, fit on training facts both models know):
  ridge           our map
  rotation        orthogonal Procrustes (semi-orthogonal across widths) + one least-squares scale
Controls:
  self            the recipient's own state for fact i (ceiling)
  token_id        ridge from the donor's EMBEDDING of the last subject token (layer 0: pure token identity)
  subject_bag     ridge from the mean donor embedding over all subject tokens (which tokens, no processing)
  random_rotation a random semi-orthogonal map with the same means and scale as `rotation`
  other_fact      the recipient's own state for a random third fact (chance of hitting answer i)
  ridge_chain / rotation_chain   (Qwen-1.5B -> SmolLM2-135M only) the three step maps composed along the chain
Layer: same relative depth in both models, chosen per method on validation facts; reported on test facts.
No training: forward passes and closed-form fits only.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/fact_swap.py
"""
import gc
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightmap import linalg
from weightmap.models import default_device, empty_cache

HERE = Path(__file__).parent
OUT = HERE / "fact_swap_results.json"
# the project's own models: the chain (Qwen-1.5B -> Qwen-0.5B -> SmolLM2-360M -> SmolLM2-135M) plus Gemma-3-270M
MODELS = {"qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
          "smol360m": "HuggingFaceTB/SmolLM2-360M", "smol135m": "HuggingFaceTB/SmolLM2-135M",
          "gemma270m": "google/gemma-3-270m"}
CHAIN = ["qwen1.5b", "qwen0.5b", "smol360m", "smol135m"]
PAIRS = [("qwen1.5b", "qwen0.5b"), ("qwen0.5b", "smol360m"), ("smol360m", "smol135m"),   # chain steps
         ("qwen1.5b", "smol135m"),                                                          # direct long jump (+ composed)
         ("qwen1.5b", "gemma270m"),                                                         # another family
         ("qwen0.5b", "qwen1.5b")]                                                          # small -> large
DEPTHS = [0.1, 0.2, 0.3, 0.4, 0.5]
PER_REL, BS = 200, 32
GRID = (1e-3, 1e-2, 1e-1, 1.0)
METHODS = ["self", "ridge", "rotation", "token_id", "subject_bag", "random_rotation", "other_fact"]
CHAIN_METHODS = ["ridge_chain", "rotation_chain"]     # only for the qwen1.5b -> smol135m pair
DEV = default_device()
rng = random.Random(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- facts
ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
by_rel = {}
for r in ds:
    by_rel.setdefault(r["relation_id"], []).append(r)
facts = []
for rel in sorted(by_rel):
    rows = by_rel[rel]
    rng.shuffle(rows)
    facts += [{"rel": rel, "prompt": r["prompt"], "subject": r["subject"].strip(), "answer": r["target_true"]}
              for r in rows[:PER_REL] if r["subject"].strip() in r["prompt"]]
for i, f in enumerate(facts):
    f["id"] = i
subjects = sorted({f["subject"] for f in facts})
rng.shuffle(subjects)
n_s = len(subjects)
part = {s: ("train" if k < 0.6 * n_s else "val" if k < 0.75 * n_s else "test") for k, s in enumerate(subjects)}
log(len(facts), "facts,", len(by_rel), "relations")


# ---------------------------------------------------------------- per-model pass: knows + keys
@torch.no_grad()
def model_pass(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    n = model.config.num_hidden_layers
    layers = sorted({round(d * n) for d in DEPTHS})
    ans = torch.tensor([tok(f["answer"], add_special_tokens=False).input_ids[0] for f in facts])
    pos, bag_rows = [], []
    for f in facts:
        enc = tok(f["prompt"], return_offsets_mapping=True, add_special_tokens=True)
        start = f["prompt"].index(f["subject"]); end = start + len(f["subject"])
        toks = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end and b > a]
        pos.append(toks[-1]); bag_rows.append(toks)
    keys = {l: torch.zeros(len(facts), model.config.hidden_size, dtype=torch.float16) for l in [0] + layers}
    bag = torch.zeros(len(facts), model.config.hidden_size, dtype=torch.float16)
    pred = torch.zeros(len(facts), dtype=torch.long)
    for i in range(0, len(facts), BS):
        chunk = facts[i:i + BS]
        enc = tok([f["prompt"] for f in chunk], return_tensors="pt", padding=True).to(DEV)
        out = model(**enc, output_hidden_states=True)
        last = enc["attention_mask"].sum(1) - 1
        pred[i:i + len(chunk)] = out.logits[torch.arange(len(chunk)), last].argmax(-1).cpu()
        for b in range(len(chunk)):
            for l in keys:
                keys[l][i + b] = out.hidden_states[l][b, pos[i + b]].half().cpu()
            bag[i + b] = out.hidden_states[0][b, bag_rows[i + b]].float().mean(0).half().cpu()
        del out
    empty_cache(torch.device(DEV))
    return {"tok": tok, "model": model, "n": n, "layers": layers, "ans": ans, "pos": torch.tensor(pos),
            "knows": (pred == ans), "keys": keys, "bag": bag}


M = {}
for k in {m for p in PAIRS for m in p}:
    M[k] = model_pass(MODELS[k])
    log(k, "knows", int(M[k]["knows"].sum()), "of", len(facts), "layers", M[k]["layers"])
    M[k]["model"].cpu()
    empty_cache(torch.device(DEV))


# ---------------------------------------------------------------- maps
def procrustes(X, Y):
    X, Y = X.double(), Y.double()
    mx, my = X.mean(0), Y.mean(0)
    U, Sv, Vt = torch.linalg.svd((X - mx).T @ (Y - my), full_matrices=False)
    R = U @ Vt
    s = float(Sv.sum() / (((X - mx) @ R) ** 2).sum())
    return mx, my, R, s


def rotation_fn(X, Y):
    mx, my, R, s = procrustes(X, Y)
    return lambda Z: my + s * ((Z.double() - mx) @ R)


def random_rotation_fn(X, Y, seed):
    mx, my, R, s = procrustes(X, Y)
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(max(R.shape), max(R.shape), generator=g, dtype=torch.float64))
    Rr = Q[:R.shape[0], :R.shape[1]]
    Rr = Rr / Rr.norm() * R.norm()
    return lambda Z: my + s * ((Z.double() - mx) @ Rr)


def ridge_fn(X, Y, Xv, Yv):
    lam = max(GRID, key=lambda l_: linalg.r2(linalg.fit_ridge(X, Y, l_), Xv, Yv))
    return linalg.fit_ridge(X, Y, lam)


def r2(pred, Y):
    Y = Y.double()
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


# ---------------------------------------------------------------- swap
@torch.no_grad()
def swap_acc(R_, layer, targets, hosts, vecs):
    """Write vecs[b] over the recipient's state at hosts[b]'s last subject token (hidden_states[layer]);
    success if the next token is targets[b]'s answer."""
    model, tok = R_["model"], R_["tok"]
    hit = 0
    for i in range(0, len(hosts), BS):
        hb, tb, vb = hosts[i:i + BS], targets[i:i + BS], vecs[i:i + BS]
        enc = tok([facts[h]["prompt"] for h in hb], return_tensors="pt", padding=True).to(DEV)
        p = R_["pos"][hb].to(DEV)
        v = vb.to(DEV, torch.bfloat16)

        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[torch.arange(len(hb), device=DEV), p] = v
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = model.model.layers[layer - 1].register_forward_hook(hook)
        try:
            logits = model(**enc).logits
        finally:
            hd.remove()
        last = enc["attention_mask"].sum(1) - 1
        nxt = logits[torch.arange(len(hb), device=DEV), last].argmax(-1).cpu()
        hit += int((nxt == R_["ans"][tb]).sum())
    return hit / max(1, len(hosts))


def make_pairs(ids, R_, seed):
    """For each target fact i: a host fact j (same relation, recipient knows it, different subject and answer
    token) and a random third fact k (for the other_fact control)."""
    r = random.Random(seed)
    known_R = R_["knows"]
    pool = {}
    for f in facts:
        if known_R[f["id"]]:
            pool.setdefault(f["rel"], []).append(f["id"])
    T, H, O = [], [], []
    for i in ids:
        cands = [j for j in pool.get(facts[i]["rel"], []) if j != i and facts[j]["subject"] != facts[i]["subject"]
                 and R_["ans"][j] != R_["ans"][i]]
        if not cands:
            continue
        others = [k for k in cands if R_["ans"][k] != R_["ans"][i]]
        T.append(i); H.append(r.choice(cands)); O.append(r.choice(others))
    return torch.tensor(T), torch.tensor(H), torch.tensor(O)


def step_maps(a, b, d):
    """Ridge and rotation maps a -> b at relative depth d, fit on training facts both models know."""
    la, lb = round(d * M[a]["n"]), round(d * M[b]["n"])
    ok = (M[a]["knows"] & M[b]["knows"]).nonzero().squeeze(-1).tolist()
    tr = torch.tensor([i for i in ok if part[facts[i]["subject"]] == "train"])
    va = torch.tensor([i for i in ok if part[facts[i]["subject"]] == "val"])
    X, Y = M[a]["keys"][la].float(), M[b]["keys"][lb].float()
    return ridge_fn(X[tr], Y[tr], X[va], Y[va]), rotation_fn(X[tr], Y[tr])


res = {"setup": {"models": MODELS, "pairs": PAIRS, "depths": DEPTHS, "per_relation": PER_REL,
                 "n_facts": len(facts)}, "knows": {k: int(v["knows"].sum()) for k, v in M.items()}, "pairs": {}}
for dn, rn in PAIRS:
    t0 = time.time()
    D, R_ = M[dn], M[rn]
    R_["model"].to(DEV)
    # both models know it; skip facts whose subject key would sit on position 0 (the attention sink)
    both = (D["knows"] & R_["knows"] & (D["pos"] > 0) & (R_["pos"] > 0)).nonzero().squeeze(-1).tolist()
    split = {s: torch.tensor([i for i in both if part[facts[i]["subject"]] == s]) for s in ("train", "val", "test")}
    pairs = {s: make_pairs(split[s].tolist(), R_, 1 + ("val", "test").index(s)) for s in ("val", "test")}
    out = {"n_known_both": len(both), "n_train": len(split["train"]), "n_val": len(pairs["val"][0]),
           "n_test": len(pairs["test"][0]), "by_depth": {}}
    tr, va = split["train"], split["val"]
    for d, ld, lr_ in zip(DEPTHS, [round(x * D["n"]) for x in DEPTHS], [round(x * R_["n"]) for x in DEPTHS]):
        Xd, Yr = D["keys"][ld].float(), R_["keys"][lr_].float()
        X0, Xb = D["keys"][0].float(), D["bag"].float()
        fn = {"ridge": ridge_fn(Xd[tr], Yr[tr], Xd[va], Yr[va]),
              "rotation": rotation_fn(Xd[tr], Yr[tr]),
              "token_id": ridge_fn(X0[tr], Yr[tr], X0[va], Yr[va]),
              "subject_bag": ridge_fn(Xb[tr], Yr[tr], Xb[va], Yr[va]),
              "random_rotation": random_rotation_fn(Xd[tr], Yr[tr], 7)}
        src = {"ridge": Xd, "rotation": Xd, "token_id": X0, "subject_bag": Xb, "random_rotation": Xd}
        methods = METHODS
        if (dn, rn) == (CHAIN[0], CHAIN[-1]):
            steps = [step_maps(u, w, d) for u, w in zip(CHAIN, CHAIN[1:])]
            fn["ridge_chain"] = lambda Z, st=steps: st[2][0](st[1][0](st[0][0](Z).float()).float())
            fn["rotation_chain"] = lambda Z, st=steps: st[2][1](st[1][1](st[0][1](Z).float()).float())
            src["ridge_chain"] = src["rotation_chain"] = Xd
            methods = METHODS + CHAIN_METHODS
        row = {"donor_layer": ld, "recipient_layer": lr_, "key_r2_test": {}}
        te_all = split["test"]
        for m in [x for x in methods if x not in ("self", "other_fact", "random_rotation")]:
            row["key_r2_test"][m] = round(r2(fn[m](src[m][te_all]), Yr[te_all]), 4)
        for s in ("val", "test"):
            T, H, O = pairs[s]
            accs = {}
            for m in methods:
                if m == "self":
                    v = Yr[T]
                elif m == "other_fact":
                    v = Yr[O]
                else:
                    v = fn[m](src[m][T]).float()
                accs[m] = round(swap_acc(R_, lr_, T, H, v), 4)
            row[s] = accs
        out["by_depth"][d] = row
        log(f"{dn}->{rn} depth {d} (L{ld}->L{lr_})", json.dumps({k: row[k] for k in ("key_r2_test", "val", "test")}))
    ms = list(out["by_depth"][DEPTHS[0]]["val"])
    best = {m: max(DEPTHS, key=lambda d_: out["by_depth"][d_]["val"][m]) for m in ms}
    out["test_at_best_val_depth"] = {m: {"depth": best[m], "acc": out["by_depth"][best[m]]["test"][m]} for m in ms}
    sf = out["test_at_best_val_depth"]["self"]["acc"]
    out["share_of_self"] = {m: round(v["acc"] / sf, 3) if sf else None for m, v in out["test_at_best_val_depth"].items()}
    res["pairs"][f"{dn}->{rn}"] = out
    log(f"{dn}->{rn} TEST", json.dumps(out["test_at_best_val_depth"]), "share_of_self", json.dumps(out["share_of_self"]),
        f"[{time.time() - t0:.0f}s]")
    R_["model"].cpu()
    gc.collect(); empty_cache(torch.device(DEV))
    OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
