"""Score a tiny model on the exp1 held-out problems so the Tiny/Small/Large Venn has a real third circle.

Same prompts, greedy decoding, token budgets and grader as exp1 stage 1 (scripts/exp1_qwen_0.5b_1.5b/s1_generate.py).
Tiny = SmolLM2-135M-Instruct: the base SmolLM2-135M used in the loop experiments has no chat template, so it
cannot take the same chat prompt. Forward passes only.

Run:  caffeinate -i env HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 uv run python scripts/tiny_venn.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "exp1_qwen_0.5b_1.5b"))
from common import (OUT, correct, generate, last_boxed, load, log, max_new, prompt_text, read, save,  # noqa: E402
                    split_data)
from math_verify import parse, verify  # noqa: E402


def lenient_math(text, gold):
    """Boxed answer if present, else the last number. Tiny never writes \\boxed{}, so the strict grader scores
    it 0 for format alone; this grader is applied to all three models in the MATH Venn."""
    b = last_boxed(text)
    if b is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
        if not nums:
            return False
        b = nums[-1]
    try:
        return bool(verify(parse("$" + gold + "$"), parse("$" + b + "$")))
    except Exception:
        return False

T_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
RES = HERE / "tiny_venn_results.json"

tok, model = load(T_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
summary = {"tiny": T_NAME}
for task in ("math", "nq"):
    f = f"gen_T_{task}_test.json"
    if not (OUT / f).exists():
        data = split_data(task, "test")
        log("T", task, len(data))
        ps = [prompt_text(tok, task, d["q"]) for d in data]
        outs = generate(tok, model, ps, max_new=max_new(task), bs=32 if task == "math" else 100)
        save(f, [{"id": d["id"], "out": o[0], "ok": correct(task, o[0], d["gold"])} for d, o in zip(data, outs)])
    T = {r["id"]: r["ok"] for r in read(f)}
    S = {r["id"]: r["ok"] for r in read(f"gen_S_{task}_test.json")}
    L = {r["id"]: r["ok"] for r in read(f"gen_L_{task}_test.json")}
    regions = {}
    for i in T:
        key = "".join(c for c, ok in zip("TSL", (T[i], S[i], L[i])) if ok) or "none"
        regions[key] = regions.get(key, 0) + 1
    summary[task] = {"n": len(T), "acc": {k: round(sum(v.values()) / len(v), 4) for k, v in (("T", T), ("S", S), ("L", L))},
                     "regions": dict(sorted(regions.items()))}
    if task == "math":
        gold = {d["id"]: d["gold"] for d in split_data(task, "test")}
        G = {tag: {r["id"]: lenient_math(r["out"], gold[r["id"]]) for r in read(f"gen_{tag}_math_test.json")}
             for tag in "TSL"}
        reg = {}
        for i in G["T"]:
            key = "".join(c for c in "TSL" if G[c][i]) or "none"
            reg[key] = reg.get(key, 0) + 1
        summary["math_lenient"] = {"acc": {t: round(sum(G[t].values()) / len(G[t]), 4) for t in "TSL"},
                                   "regions": dict(sorted(reg.items()))}
    log(task, json.dumps(summary[task]))
RES.write_text(json.dumps(summary, indent=1))
log("wrote", RES)
