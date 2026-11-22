"""Extend the model atlas (scripts/atlas_recursion.py) with more small models, then test the
capability <-> map-quality correlation with an exact permutation p-value. Same texts, same settings.
batch_size=1 everywhere so absolute-position models (GPT-2) see no left padding.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/atlas_extend.py
"""
import gc
import itertools
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightmap import WeightMap, align
from weightmap.models import default_device, empty_cache

HERE = Path(__file__).parent
EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
PREV = HERE / "atlas_recursion_results.json"
OUT = HERE / "atlas_extend_results.json"
LARGE = "Qwen/Qwen2.5-1.5B-Instruct"
NEW = ["Qwen/Qwen2.5-0.5B", "gpt2", "gpt2-medium", "google/gemma-3-1b-pt", "llmware/bling-1b-0.1"]
DEV = default_device()
torch.manual_seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    return model, tok


texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
texts = [t for t in texts if len(t.strip()) > 20]


@torch.no_grad()
def bits_per_byte(model, tok, texts, max_length=512):
    """Same as atlas_recursion.bits_per_byte, one text at a time (no padding)."""
    nll, nbytes = 0.0, 0
    for t in texts:
        enc, offs = align.tokenize(tok, [t], max_length)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        lp = torch.log_softmax(model(**enc).logits.float()[:, :-1], -1)
        nll -= float(lp.gather(-1, enc["input_ids"][:, 1:].unsqueeze(-1)).sum())
        o = offs[0]
        nbytes += len(t[int(o[1:, 0].min()):int(o[:, 1].max())].encode("utf-8")) if len(o) > 1 else 0
        del lp
    empty_cache(torch.device(DEV))
    return nll / math.log(2) / max(nbytes, 1)


def spearman_exact(x, y):
    """Spearman rho and exact two-sided permutation p (n <= 10 is enumerable)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    def rho(p):
        return 1 - 6 * sum((rx[i] - p[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    obs = rho(ry)
    perms = list(itertools.permutations(range(n))) if n <= 10 else None
    hits = sum(abs(rho(p)) >= abs(obs) - 1e-12 for p in perms)
    return round(obs, 4), round(hits / len(perms), 4)


prev = json.loads(PREV.read_text())
res = {"large": prev["large"], "atlas": list(prev["atlas"]), "added": [], "skipped": {}}
if OUT.exists():                                      # resume
    res = json.loads(OUT.read_text())
L, tokL = load(LARGE)
for name in NEW:
    if name in res["added"] or name in res["skipped"]:
        continue
    t0 = time.time()
    try:
        S, tokS = load(name)
    except Exception as e:                            # noqa: BLE001 - record and move on
        res["skipped"][name] = repr(e)[:300]
        log("SKIP", name, repr(e)[:200])
        continue
    bpb = bits_per_byte(S, tokS, texts[:200])
    wm = WeightMap(large=(L, tokL), small=(S, tokS))
    layers = sorted({round(0.4 * wm.n_small), round(0.6 * wm.n_small)})
    wm.fit(texts, layers=layers, positions_per_text=32, batch_size=1, max_length=512, cka_texts=80,
           max_tokens=12000)
    rows = wm.report()
    entry = {"small": name, "n_layers": wm.n_small + 1, "shared_vocab": wm.shared_vocab,
             "params_M": round(sum(p.numel() for p in S.parameters()) / 1e6, 1),
             "bits_per_byte": round(bpb, 4),
             "mean_r2_L_to_S": round(sum(r["r2_large_to_small"] for r in rows) / len(rows), 4),
             "mean_r2_S_to_L": round(sum(r["r2_small_to_large"] for r in rows) / len(rows), 4),
             "layers": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
             "run": "atlas_extend (batch_size=1)"}
    res["atlas"].append(entry); res["added"].append(name)
    log(f"ATLAS {name} [{time.time() - t0:.0f}s]",
        {k: entry[k] for k in ("params_M", "bits_per_byte", "mean_r2_L_to_S", "mean_r2_S_to_L")})
    del wm, S
    gc.collect(); empty_cache(torch.device(DEV))
    OUT.write_text(json.dumps(res, indent=1))

at = res["atlas"]
bpb = [-e["bits_per_byte"] for e in at]               # higher = more capable
stats = {"n_models": len(at)}
for key in ("mean_r2_S_to_L", "mean_r2_L_to_S"):
    stats[key] = dict(zip(("spearman_rho", "p_exact"), spearman_exact(bpb, [e[key] for e in at])))
first5 = at[:5]
stats["original_5_S_to_L"] = dict(zip(("spearman_rho", "p_exact"),
                                      spearman_exact([-e["bits_per_byte"] for e in first5],
                                                     [e["mean_r2_S_to_L"] for e in first5])))
res["correlation"] = stats
log("CORRELATION", json.dumps(stats))
OUT.write_text(json.dumps(res, indent=1))
log("wrote", OUT)
