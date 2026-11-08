"""Stage 4: training-free edits of S, evaluated on held-out test problems.

Conditions (all applied to S only):
  base       : unmodified S
  S-native   : add  c*|d_S| * unit(d_S)          at S layer a   (S's own correct-vs-wrong direction)
  L->S       : add  c*|d_S| * unit(Map(d_L))     at S layer a   (L's direction projected into S)
  random     : add  c*|d_S| * unit(random)       (same layer/scale as best L->S)
  transplant : per problem, mix L's mapped prompt states into S's: h <- (1-b) h + b Map(h_L)   (needs L at runtime)
Tune (layer, c / b) on a tune slice, report on the disjoint test slice.
"""
import numpy as np
from common import *

CS = [1, 2, 4]   # multiples of the natural size |d_S| of the correct-vs-wrong difference
BETAS = [0.5, 1.0]
TUNE_N = {"math": 40, "nq": 100}

tokS, S = load(S_NAME)
tokL, L = load(L_NAME)
dirs = torch.load(OUT / "dirs.pt", weights_only=False)
maps = torch.load(OUT / "maps.pt", weights_only=False)
match = {int(k): v for k, v in read("layer_match.json").items()}


def unit(v):
    v = v.float()
    return v / v.norm()


def vec_for(task, method, a, c):
    d = dirs[task][a]
    if method == "random":
        v = torch.randn(d["S"].shape, generator=torch.Generator().manual_seed(1234 + a))
    else:
        v = d["S" if method == "S-native" else "LtoS"]
    return (unit(v) * c * d["S"].float().norm()).to(DEV)   # same norm for every method


def transplant_ctx(a, beta):
    W, b = maps[a]
    W, b = W.float().to(DEV), b.float().to(DEV)
    lL = match[a]

    def make(enc):
        hL = hidden_states(L, enc)[lL].float()
        target = (hL @ W + b)
        plen = enc.input_ids.shape[1]

        def fn(h):
            if h.shape[1] != plen:      # decoding steps: untouched
                return h
            n = h.float().norm(dim=-1, keepdim=True)
            sink = n > 5 * n.median()   # don't overwrite attention-sink / massive-activation tokens
            new = (1 - beta) * h.float() + beta * target
            return torch.where(sink, h.float(), new).to(h.dtype)
        return hook_layer(S, a, fn)
    return make


def run(task, items, cond):
    """cond: ('base',) | (method, a, c) | ('transplant', a, beta). Returns list of 0/1, cached."""
    key = f"{task}|{'|'.join(map(str, cond))}|{len(items)}|{items[0]['id']}"
    cache_f = OUT / "s4_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    if key in cache:
        return cache[key]
    ps = [prompt_text(tokS, task, d["q"]) for d in items]
    bs = 32 if task == "math" else 100
    if cond[0] == "base":
        outs = generate(tokS, S, ps, max_new=max_new(task), bs=bs)
    elif cond[0] == "transplant":
        outs = generate(tokS, S, ps, max_new=max_new(task), bs=bs, prefill_fn=transplant_ctx(cond[1], cond[2]))
    else:
        with add_vec(S, cond[1], vec_for(task, *cond)):
            outs = generate(tokS, S, ps, max_new=max_new(task), bs=bs)
    oks = [int(correct(task, o[0], d["gold"])) for d, o in zip(items, outs)]
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    cache[key] = oks
    cache_f.write_text(json.dumps(cache))
    log(f"  {key}: acc={np.mean(oks):.3f}")
    return oks


def boot_ci(x, y, n=5000):
    x, y = np.array(x), np.array(y)
    d = y - x
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


report = {}
for task in ["math", "nq"]:
    data = split_data(task, "test")
    S1 = {r["id"]: r["ok"] for r in read(f"gen_S_{task}_test.json")}
    L1 = {r["id"]: r["ok"] for r in read(f"gen_L_{task}_test.json")}
    lat = {int(k): v for k, v in read(f"latent_{task}.json").items()}
    order = random.Random(7).sample(data, len(data))
    tune, test = order[:TUNE_N[task]], order[TUNE_N[task]:]
    log(task, "tune", len(tune), "test", len(test))

    base_t = np.mean(run(task, tune, ("base",)))
    best = {}
    for method in ["S-native", "L->S"]:
        grid = [(method, a, c) for a in S_LAYERS for c in CS]
        scores = {g: np.mean(run(task, tune, g)) for g in grid}
        best[method] = max(scores, key=scores.get)
    grid = [("transplant", a, b) for a in S_LAYERS for b in BETAS]
    scores = {g: np.mean(run(task, tune, g)) for g in grid}
    best["transplant"] = max(scores, key=scores.get)
    best["random"] = ("random", best["L->S"][1], best["L->S"][2])
    log(task, "tune base", base_t, "best configs", best)

    res = {"base": run(task, test, ("base",))}
    for m, cfg in best.items():
        res[m] = run(task, test, cfg)
    res["L (upper bound)"] = [int(L1[d["id"]]) for d in test]

    groups = {
        "all": [i for i, _ in enumerate(test)],
        "target-blocked (L ok, S wrong, S pass@8>0)": [i for i, d in enumerate(test) if d["id"] in lat and lat[d["id"]] > 0],
        "target-absent (L ok, S wrong, S pass@8=0)": [i for i, d in enumerate(test) if d["id"] in lat and lat[d["id"]] == 0],
        "S-already-correct (damage check)": [i for i, d in enumerate(test) if S1[d["id"]]],
    }
    tab = {}
    for gname, ix in groups.items():
        row = {"n": len(ix)}
        for m, oks in res.items():
            sub = [oks[i] for i in ix]
            row[m] = round(float(np.mean(sub)), 3) if ix else None
        tab[gname] = row
    cis = {m: boot_ci(res["base"], res[m]) for m in res if m not in ("base",)}
    report[task] = {"best_configs": {k: list(v) for k, v in best.items()}, "table": tab,
                    "delta_vs_base_all (mean, 95% CI)": cis}
    log(json.dumps(report[task], indent=1))
save("s4_report.json", report)
