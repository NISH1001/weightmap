"""One-off check that the `weightmap` package reproduces experiment 1's map quality on real models,
and that the cross-tokenizer path works on a genuinely different model family.

Run:  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
      uv run python scripts/validate_library.py
"""
import json
import time
from pathlib import Path

from weightmap import WeightMap

EXP1 = Path(__file__).parent / "exp1_qwen_0.5b_1.5b" / "results"
OUT = Path(__file__).parent / "validate_library_results.json"

# fit texts: the large model's own outputs from experiment 1 (MATH + NQ train splits)
texts = [r["out"] for f in ("gen_L_math_train.json", "gen_L_nq_train.json")
         for r in json.loads((EXP1 / f).read_text())]
print(len(texts), "fit texts")

FIT = dict(positions_per_text=40, batch_size=4, max_length=768, cka_texts=150, max_tokens=15000)
results = {}

t = time.time()
wm = WeightMap(large="Qwen/Qwen2.5-1.5B-Instruct", small="Qwen/Qwen2.5-0.5B-Instruct")
wm.fit(texts, layers=[10, 14], **FIT)
results["qwen0.5b<-qwen1.5b (shared tokenizer)"] = wm.report()
print(f"shared tokenizer [{time.time() - t:.0f}s]")
for r in wm.report():
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
del wm

t = time.time()
wm = WeightMap(large="Qwen/Qwen2.5-1.5B-Instruct", small="HuggingFaceTB/SmolLM2-360M")
print("SmolLM2-360M layers:", wm.n_small + 1, "shared vocab:", wm.shared_vocab)
mid = [wm.n_small // 3, wm.n_small // 2, (2 * wm.n_small) // 3]
wm.fit(texts, layers=mid, **FIT)
results["smollm2-360m<-qwen1.5b (different tokenizers)"] = wm.report()
print(f"cross tokenizer [{time.time() - t:.0f}s]")
for r in wm.report():
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
print("best L layer per SmolLM2 layer:", wm.layer_match)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)
