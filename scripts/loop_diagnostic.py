"""Can a large model's view diagnose a small model's failure? (No training; steering only.)

S = SmolLM2-135M (base), which falls into repetition loops on GSM8K. L = Qwen2.5-1.5B-Instruct, which doesn't.
1. Generate S on GSM8K *train* prompts; label each output looping / productive with the repetition detector
   (Barry-style: 16-token windows; loop = suffix where >=80% of tokens repeat, >=64 tokens long).
2. "Loop direction" = mean state over loop regions - mean state over productive completions, built two ways
   on the SAME texts:  S-native (S encodes them)  vs  L->S (Qwen encodes them, mapped into S via WeightMap,
   cross-tokenizer).  Plus a random direction of the same norm.
3. Steer S *against* each direction; tune (layer, strength) on a tune slice, report on held-out GSM8K test.
Metrics: loop rate (primary), GSM8K accuracy, and early coherence = Qwen-1.5B NLL/token of the first
48 completion tokens (before most loops begin; loops themselves are low-NLL so full-text NLL would reward them).

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/loop_diagnostic.py
"""
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset

from weightmap import WeightMap
from weightmap.models import empty_cache, model_device

HERE = Path(__file__).parent
# FIXED="18:2" skips tuning and evaluates every method at that (layer, scale) - chosen from the tune grid
FIXED = os.environ.get("FIXED")
OUT = HERE / ("loop_diagnostic_results.json" if not FIXED else f"loop_diagnostic_fixed_{FIXED.replace(':', 'x')}.json")
S_NAME, L_NAME = "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW, W = 256, 16
N_DIR, N_TUNE, N_TEST = 400, 60, 200
LAYERS = [12, 18]                      # SmolLM2-135M has 30 layers: ~40% and ~60% depth
SCALES = [1, 2, 4]
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
    """Index of the first token of the loop region, or None."""
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
tune, test = test_all.select(range(N_TUNE)), test_all.select(range(N_TUNE, N_TUNE + N_TEST))

wm = WeightMap(large=L_NAME, small=S_NAME)
tokS = wm.tok_small
log("models loaded; S layers:", wm.n_small + 1, "shared vocab:", wm.shared_vocab)


def run(ds, steer=None):
    ps = [prompt(r["question"]) for r in ds]
    outs = wm.generate(ps, max_new_tokens=MAX_NEW, steer=steer, batch_size=40)
    loops = [loop_start(tokS(o, add_special_tokens=False).input_ids) is not None for o in outs]
    acc = [answer(o) == gold(r["answer"]) for o, r in zip(outs, ds)]
    return ps, outs, loops, acc


@torch.no_grad()
def early_nll(ps, outs, n_tok=48):
    """Qwen-1.5B NLL per token of the first n_tok completion tokens (conditioned on the prompt)."""
    tokL, L = wm.tok_large, wm.large
    dev = model_device(L)
    tot, cnt = 0.0, 0
    for p, o in zip(ps, outs):
        comp = tokL(o, add_special_tokens=False).input_ids[:n_tok]
        if not comp:
            continue
        pid = tokL(p).input_ids
        ids = torch.tensor([pid + comp], device=dev)
        lp = torch.log_softmax(L(ids).logits.float()[0, len(pid) - 1:-1], -1)
        tot -= float(lp.gather(-1, torch.tensor(comp, device=dev).unsqueeze(-1)).sum()); cnt += len(comp)
    empty_cache(dev)
    return tot / max(cnt, 1)


def summarize(name, ps, outs, loops, acc, nll=True):
    r = {"loop_rate": round(sum(loops) / len(loops), 4), "accuracy": round(sum(acc) / len(acc), 4),
         "mean_chars": round(sum(len(o) for o in outs) / len(outs), 1)}
    if nll:
        r["early_nll_qwen"] = round(early_nll(ps, outs), 4)
    log(name, r)
    return r


res = {"setup": {"S": S_NAME, "L": L_NAME, "max_new_tokens": MAX_NEW, "n_dir": N_DIR, "n_tune": N_TUNE,
                 "n_test": N_TEST, "layers": LAYERS, "scales": SCALES}}

# ---------------------------------------------------------------- 1. label S's own outputs (train prompts)
ps, outs, loops, acc = run(train)
res["train_base"] = summarize("train base", ps, outs, loops, acc, nll=False)
loop_texts, loop_start_chars, prod_texts, prod_start_chars = [], [], [], []
for p, o in zip(ps, outs):
    ids = tokS(o, add_special_tokens=False).input_ids
    st = loop_start(ids)
    if st is not None:
        prefix = tokS.decode(ids[:st], skip_special_tokens=True)
        loop_texts.append(p + o); loop_start_chars.append(len(p) + len(prefix))
    else:
        prod_texts.append(p + o); prod_start_chars.append(len(p))
log("looping texts:", len(loop_texts), "productive texts:", len(prod_texts))
res["n_loop_texts"], res["n_prod_texts"] = len(loop_texts), len(prod_texts)

# ---------------------------------------------------------------- 2. map + directions
fit_texts = [p + o for p, o in zip(ps, outs)]
wm.fit(fit_texts, layers=LAYERS, positions_per_text=32, batch_size=4, max_length=512, cka_texts=100,
       max_tokens=12000)
res["map"] = wm.report()
for r in wm.report():
    log("map", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

dirs = {}
for s in LAYERS:
    l = wm.layer_match[s]
    dS = (wm.encode(loop_texts, s, "small", loop_start_chars).mean(0)
          - wm.encode(prod_texts, s, "small", prod_start_chars).mean(0))
    dL = (wm.encode(loop_texts, l, "large", loop_start_chars).mean(0)
          - wm.encode(prod_texts, l, "large", prod_start_chars).mean(0))
    dLS = wm.transport(dL, s)
    cos = float(torch.nn.functional.cosine_similarity(dS, dLS, dim=0))
    dirs[s] = {"S-native": dS, "L->S": dLS, "random": torch.randn_like(dS), "norm": float(dS.norm())}
    res.setdefault("direction_cosine", {})[s] = round(cos, 4)
    log(f"layer {s} (L{l}): cos(d_S, Map(d_L)) = {cos:.3f}, |d_S| = {dS.norm():.2f}")


def vec(method, s, c):
    d = dirs[s][method]
    return -(d / d.norm()) * c * dirs[s]["norm"]          # steer AGAINST the loop direction


# ---------------------------------------------------------------- 3. tune (or fixed config)
if FIXED:
    fl, fc = (int(x) for x in FIXED.split(":"))
    best = {m: (fl, fc) for m in ("S-native", "L->S", "random")}
    res["best_configs"] = {k: list(v) for k, v in best.items()}
    res["note"] = f"fixed config {FIXED} chosen from the tune grid of the main run (same for every method)"
    res["test"] = {"base": summarize("TEST base", *run(test))}
    for method, (s_, c_) in best.items():
        res["test"][method] = summarize(f"TEST {method} L{s_} x{c_}", *run(test, steer=(vec(method, s_, c_), s_)))
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log("wrote", OUT)
    raise SystemExit
ps, outs, loops, acc = run(tune)
base_tune = summarize("tune base", ps, outs, loops, acc)
res["tune"] = {"base": base_tune}
best = {}
for method in ("S-native", "L->S"):
    cands = []
    for s in LAYERS:
        for c in SCALES:
            r = summarize(f"tune {method} L{s} x{c}", *run(tune, steer=(vec(method, s, c), s)))
            res["tune"][f"{method}|{s}|{c}"] = r
            # lowest loop rate whose early coherence is no worse than base + 15%
            if r["early_nll_qwen"] <= base_tune["early_nll_qwen"] * 1.15:
                cands.append((r["loop_rate"], -r["accuracy"], s, c))
    best[method] = min(cands)[2:] if cands else (LAYERS[0], SCALES[0])
best["random"] = best["L->S"]
res["best_configs"] = {k: list(v) for k, v in best.items()}
log("best configs", best)

# ---------------------------------------------------------------- 4. held-out test
res["test"] = {"base": summarize("TEST base", *run(test))}
for method, (s, c) in best.items():
    res["test"][method] = summarize(f"TEST {method} L{s} x{c}", *run(test, steer=(vec(method, s, c), s)))
OUT.write_text(json.dumps(res, indent=1, default=float))
log("wrote", OUT)
