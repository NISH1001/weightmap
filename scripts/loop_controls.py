"""Controls for the loop-direction result of scripts/loop_diagnostic.py.

Concern: there the map was fitted on the same SmolLM2 outputs that define the "looping" and "productive" groups,
so the mapped direction could match S's own direction by construction. Three controls, same models, layers
(S block 18, L block 10), prompts, detector, steering layer and strength (2 x |d_S|) as the original:

  A1  independent map   the L->S map is fitted on unrelated text (Qwen-1.5B's MATH/NQ answers from exp1)
  A2  embedding map     the map reads only L's input embeddings (token identity, nothing L computed),
                        fitted on unrelated text; the direction is the difference of L's embedding-layer means
  A3  random split      looping/productive labels shuffled over texts: does any group difference transfer?
  B   L's own loop rate Qwen-1.5B answers the same 200 held-out prompts; same detector

Each direction is scored by cosine with S's own direction and by steering S against it on the held-out prompts.
No training: forward passes and closed-form ridge.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/loop_controls.py
"""
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset

from weightmap import WeightMap, align, linalg
from weightmap.models import base_model, empty_cache, model_device

HERE = Path(__file__).parent
OUT = HERE / "loop_controls_results.json"
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
S_NAME, L_NAME = "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-1.5B-Instruct"
S_LAYER, L_LAYER, SCALE = 18, 10, 2          # the setting of the original comparison
S_RAW, L_RAW, EMB = S_LAYER + 1, L_LAYER + 1, 0   # indices into output_hidden_states (0 = embedding output)
MAX_NEW, W = 256, 16
N_DIR, N_TUNE, N_TEST = 400, 60, 200
K, GRID = 32, (1e-4, 1e-3, 1e-2, 1e-1)
torch.manual_seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def prompt(q):
    return f"Question: {q}\nAnswer: Let's think step by step."


def gold(a):
    return a.split("####")[-1].strip().replace(",", "")


def answer(t):
    t = t.split("\nQuestion:")[0]
    m = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return m[-1].replace(",", "").rstrip(".") if m else ""


def loop_start(ids):
    """Index of the first token of the loop region, or None (identical to loop_diagnostic.py)."""
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
test_all = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(N_TUNE + N_TEST))
test = test_all.select(range(N_TUNE, N_TUNE + N_TEST))

wm = WeightMap(large=L_NAME, small=S_NAME)
S, L, tokS, tokL = wm.small, wm.large, wm.tok_small, wm.tok_large
devS, devL = model_device(S), model_device(L)
log("models loaded")


def run(ds, steer=None):
    ps = [prompt(r["question"]) for r in ds]
    outs = wm.generate(ps, max_new_tokens=MAX_NEW, steer=steer, batch_size=40)
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
        ids = torch.tensor([pid + comp], device=devL)
        lp = torch.log_softmax(L(ids).logits.float()[0, len(pid) - 1:-1], -1)
        tot -= float(lp.gather(-1, torch.tensor(comp, device=devL).unsqueeze(-1)).sum()); cnt += len(comp)
    empty_cache(devL)
    return tot / max(cnt, 1)


def summarize(name, ps, outs, loops, acc):
    r = {"loop_rate": round(sum(loops) / len(loops), 4), "accuracy": round(sum(acc) / len(acc), 4),
         "early_nll_qwen": round(early_nll(ps, outs), 4)}
    log(name, r)
    return r


@torch.no_grad()
def states(model, enc, dev):
    return base_model(model)(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states


@torch.no_grad()
def collect(texts):
    """Aligned token states: S block-18 output, L block-10 output, L embedding output; text id per row."""
    XS, XL, XE, G = [], [], [], []
    for i in range(0, len(texts), 4):
        chunk = texts[i:i + 4]
        enc_s, off_s = align.tokenize(tokS, chunk, 512)
        enc_l, off_l = align.tokenize(tokL, chunk, 512)
        hs, hl = states(S, enc_s, devS), states(L, enc_l, devL)
        for j in range(len(chunk)):
            pairs = align.boundary_pairs(off_s[j], enc_s["attention_mask"][j].bool(),
                                         off_l[j], enc_l["attention_mask"][j].bool())
            if not pairs:
                continue
            if len(pairs) > K:
                g = torch.Generator().manual_seed(i + j)
                pairs = [pairs[t] for t in torch.randperm(len(pairs), generator=g)[:K].sort().values.tolist()]
            a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
            XS.append(hs[S_RAW][j, a.to(devS)].float().cpu())
            XL.append(hl[L_RAW][j, b.to(devL)].float().cpu())
            XE.append(hl[EMB][j, b.to(devL)].float().cpu())
            G.append(torch.full((len(pairs),), i + j))
        del hs, hl
        empty_cache(devS)
    XS, XL, XE, G = torch.cat(XS), torch.cat(XL), torch.cat(XE), torch.cat(G)
    ns, nl = XS.norm(dim=1), XL.norm(dim=1)
    keep = (ns < 5 * ns.median()) & (nl < 5 * nl.median())      # drop attention-sink-like rows, as the library does
    return XS[keep], XL[keep], XE[keep], G[keep]


def fit_map(X, Y, G):
    """Ridge X -> Y; lambda on an inner text-level split; held-out R^2 on outer test texts; refit on all rows."""
    tr, te = linalg.split(len(G), 0.2, 0, G)
    tr_in, va_in = (tr[t] for t in linalg.split(len(tr), 0.2, 1, G[tr]))
    lam = max(GRID, key=lambda l_: linalg.heldout_r2(X, Y, l_, tr_in, va_in))
    return linalg.fit_ridge(X, Y, lam).W, round(linalg.heldout_r2(X, Y, lam, tr, te), 4)


@torch.no_grad()
def text_means(model, tok, texts, starts, raw_idx, dev):
    """Per-text mean state over tokens starting at or after each text's start character."""
    out = []
    for i in range(0, len(texts), 4):
        chunk = texts[i:i + 4]
        enc, offs = align.tokenize(tok, chunk, 2048)
        hs = states(model, enc, dev)[raw_idx]
        for j in range(len(chunk)):
            pos = align.positions_from_char(offs[j], enc["attention_mask"][j].bool(), starts[i + j])
            assert pos, "no tokens after the region start"
            out.append(hs[j, pos].float().mean(0).cpu())
        del hs
        empty_cache(dev)
    return torch.stack(out).double()


def cos(a, b):
    return round(float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0)), 4)


res = {"setup": {"S": S_NAME, "L": L_NAME, "S_layer": S_LAYER, "L_layer": L_LAYER, "scale": SCALE,
                 "n_dir": N_DIR, "n_test": N_TEST}}

# ---------------------------------------------------------------- 1. S's outputs on the training prompts, labelled
ps, outs, loops, _ = run(train)
texts, starts, is_loop = [], [], []
for p, o in zip(ps, outs):
    ids = tokS(o, add_special_tokens=False).input_ids
    st = loop_start(ids)
    texts.append(p + o)
    starts.append(len(p) if st is None else len(p) + len(tokS.decode(ids[:st], skip_special_tokens=True)))
    is_loop.append(st is not None)
is_loop = torch.tensor(is_loop)
res["n_loop_texts"], res["n_prod_texts"] = int(is_loop.sum()), int((~is_loop).sum())
log("loop texts", res["n_loop_texts"], "productive", res["n_prod_texts"])

# ---------------------------------------------------------------- 2. maps: same texts vs unrelated texts
other = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json") for r in json.loads((EXP1 / f).read_text())]
other = [t for t in other if len(t.strip()) > 20]
maps = {}
for name, tx in (("same_texts", texts), ("unrelated_texts", other)):
    XS, XL, XE, G = collect(tx)
    W_l, r2_l = fit_map(XL, XS, G)
    W_e, r2_e = fit_map(XE, XS, G)
    maps[name] = {"W": W_l, "We": W_e}
    res.setdefault("maps", {})[name] = {"n_texts": len(tx), "n_tokens": len(G), "r2_L10_to_S18": r2_l,
                                        "r2_Lembedding_to_S18": r2_e}
    log("map", name, res["maps"][name])
    del XS, XL, XE

# ---------------------------------------------------------------- 3. directions
MS = text_means(S, tokS, texts, starts, S_RAW, devS)
ML = text_means(L, tokL, texts, starts, L_RAW, devL)
ME = text_means(L, tokL, texts, starts, EMB, devL)


def diff(M, lab):
    return M[lab].mean(0) - M[~lab].mean(0)


dS, dL, dE = diff(MS, is_loop), diff(ML, is_loop), diff(ME, is_loop)
dirs = {"S_own": dS,
        "L_same_map": dL @ maps["same_texts"]["W"],
        "L_independent_map": dL @ maps["unrelated_texts"]["W"],
        "L_embedding_only_independent_map": dE @ maps["unrelated_texts"]["We"],
        "L_embedding_only_same_map": dE @ maps["same_texts"]["We"]}
res["cosine_with_S_own"] = {k: cos(dS, v) for k, v in dirs.items() if k != "S_own"}
res["norm_S_own"] = round(float(dS.norm()), 3)
log("cosine with S's own direction", res["cosine_with_S_own"])

# A3: shuffled labels. If a random split of the same texts also transfers with high cosine, the cosine reflects
# the map's general quality on these texts and is not specific to looping.
rand = {k: [] for k in ("same_map", "independent_map", "embedding_only_independent_map", "vs_true_direction")}
for seed in range(20):
    lab = is_loop[torch.randperm(len(is_loop), generator=torch.Generator().manual_seed(seed))]
    rS, rL, rE = diff(MS, lab), diff(ML, lab), diff(ME, lab)
    rand["same_map"].append(cos(rS, rL @ maps["same_texts"]["W"]))
    rand["independent_map"].append(cos(rS, rL @ maps["unrelated_texts"]["W"]))
    rand["embedding_only_independent_map"].append(cos(rS, rE @ maps["unrelated_texts"]["We"]))
    rand["vs_true_direction"].append(cos(rS, dS))
res["random_split_cosine"] = {k: {"mean": round(sum(v) / len(v), 4), "min": min(v), "max": max(v)} for k, v in rand.items()}
log("random-split cosine (20 shuffles)", res["random_split_cosine"])
OUT.write_text(json.dumps(res, indent=1))

# ---------------------------------------------------------------- 4. steer S against each direction (held-out prompts)
def vec(d):
    return (-(d / d.norm()) * SCALE * float(dS.norm())).float()


dirs["random"] = torch.randn(dS.shape, generator=torch.Generator().manual_seed(0)).double()
res["test"] = {"base": summarize("TEST base", *run(test))}
for name in ("S_own", "L_same_map", "L_independent_map", "L_embedding_only_independent_map", "random"):
    res["test"][name] = summarize(f"TEST {name}", *run(test, steer=(vec(dirs[name]), S_LAYER)))
    OUT.write_text(json.dumps(res, indent=1))

# ---------------------------------------------------------------- 4b. dose response
# The independent-map direction costs less coherence at strength 2 than S's own, so it may simply be a weaker
# push. Other strengths for both, each reported with its coherence cost (strength 2 is in "test" above).
res["dose_response"] = {}
for name in ("S_own", "L_independent_map"):
    for sc in (1, 3, 4):
        v = (-(dirs[name] / dirs[name].norm()) * sc * float(dS.norm())).float()
        res["dose_response"][f"{name} x{sc}"] = summarize(f"DOSE {name} x{sc}", *run(test, steer=(v, S_LAYER)))
        OUT.write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- 5. B: does the large model loop on these prompts?
@torch.no_grad()
def generate_large(prompts, bs=16):
    side = tokL.padding_side
    tokL.padding_side = "left"
    outs_l = []
    try:
        for i in range(0, len(prompts), bs):
            enc = tokL(prompts[i:i + bs], return_tensors="pt", padding=True).to(devL)
            g = L.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tokL.pad_token_id)
            outs_l += tokL.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            del g, enc
            empty_cache(devL)
    finally:
        tokL.padding_side = side
    return outs_l


tp = [prompt(r["question"]) for r in test]
lo = generate_large(tp)
res["large_model_on_test_prompts"] = {
    "n": len(lo),
    "loop_rate_detector_on_S_tokens": round(sum(loop_start(tokS(o, add_special_tokens=False).input_ids) is not None for o in lo) / len(lo), 4),
    "loop_rate_detector_on_L_tokens": round(sum(loop_start(tokL(o, add_special_tokens=False).input_ids) is not None for o in lo) / len(lo), 4),
    "accuracy": round(sum(answer(o) == gold(r["answer"]) for o, r in zip(lo, test)) / len(lo), 4)}
log("Qwen-1.5B on the same held-out prompts", res["large_model_on_test_prompts"])
OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
