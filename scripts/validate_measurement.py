"""One-off check of the measurement fixes.

Q1  Was experiment 1's R^2 = 0.85 inflated? 2x2 on Qwen2.5-0.5B <- 1.5B (layers 10, 14, CKA pairing
    as in exp1):  {token-level, text-level holdout} x {completions only, chat-formatted prompt+completion}.
Q2  Does R^2-based layer pairing beat CKA pairing across families? SmolLM2-360M <- Qwen2.5-1.5B,
    text-level holdout.

Run:  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
      uv run python scripts/validate_measurement.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "exp1_qwen_0.5b_1.5b"))
from common import prompt_text, task_data  # noqa: E402  (exp1 helpers: same prompts/sampling as exp1)

from weightmap import WeightMap  # noqa: E402

EXP1 = HERE / "exp1_qwen_0.5b_1.5b" / "results"
OUT = HERE / "validate_measurement_results.json"
FIT = dict(positions_per_text=40, batch_size=4, max_length=768, cka_texts=100, max_tokens=15000)
LARGE = "Qwen/Qwen2.5-1.5B-Instruct"
results = {"q1_holdout_x_template": [], "q2_pairing": []}


def rows(wm, **extra):
    out = []
    for r in wm.report():
        r = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        r.update(extra)
        out.append(r)
        print(r, flush=True)
    return out


gen = {t: json.loads((EXP1 / f"gen_L_{t}_train.json").read_text()) for t in ("math", "nq")}
data = {"math": task_data("math", "train", 120), "nq": task_data("nq", "train", 400)}

# ---------------------------------------------------------------- Q1
wm = WeightMap(large=LARGE, small="Qwen/Qwen2.5-0.5B-Instruct")
completions = [r["out"] for t in ("math", "nq") for r in gen[t]]
chat = [prompt_text(wm.tok_small, t, d["q"]) + r["out"] for t in ("math", "nq") for d, r in zip(data[t], gen[t])]
for text_kind, texts in (("completions", completions), ("chat-formatted", chat)):
    for holdout_by in ("token", "text"):
        t0 = time.time()
        wm.fit(texts, layers=[10, 14], holdout_by=holdout_by, pair_by="cka", **FIT)
        print(f"--- Q1 {text_kind} / holdout_by={holdout_by} [{time.time() - t0:.0f}s]", flush=True)
        results["q1_holdout_x_template"] += rows(wm, texts=text_kind)
del wm

# ---------------------------------------------------------------- Q2
wm = WeightMap(large=LARGE, small="HuggingFaceTB/SmolLM2-360M")
layers = [wm.n_small // 3, wm.n_small // 2, (2 * wm.n_small) // 3]
for pair_by in ("cka", "r2"):
    t0 = time.time()
    wm.fit(completions, layers=layers, holdout_by="text", pair_by=pair_by, **FIT)
    print(f"--- Q2 pair_by={pair_by} [{time.time() - t0:.0f}s]  layer_match(fitted)=",
          {a: wm.layer_match[a] for a in layers}, flush=True)
    results["q2_pairing"] += rows(wm)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)
