"""Error bars for the loop-diagnostic held-out results (no compute: reads the saved JSON).
Wilson 95% intervals per rate; two-proportion z-tests between methods. The runs share the same 200 test
prompts, so an unpaired test is conservative (per-item outcomes were not saved, so McNemar isn't possible)."""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "loop_diagnostic_fixed_18x2.json"
OUT = HERE / "loop_stats_results.json"
R = json.loads(SRC.read_text())
N = R["setup"]["n_test"]


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 4), round(c + h, 4)


def ztest(p1, p2, n):
    p = (p1 + p2) / 2
    se = math.sqrt(2 * p * (1 - p) / n)
    z = (p1 - p2) / se if se else 0.0
    return round(z, 2), round(math.erfc(abs(z) / math.sqrt(2)), 4)


res = {"source": SRC.name, "n_test": N, "rates": {}, "comparisons": {}}
for m, r in R["test"].items():
    res["rates"][m] = {k: {"rate": r[k], "wilson95": wilson(r[k], N)} for k in ("loop_rate", "accuracy")}
for a, b in [("base", "L->S"), ("base", "S-native"), ("S-native", "L->S"), ("base", "random"), ("L->S", "random")]:
    for k in ("loop_rate", "accuracy"):
        z, p = ztest(R["test"][a][k], R["test"][b][k], N)
        res["comparisons"][f"{k}: {a} vs {b}"] = {"diff": round(R["test"][a][k] - R["test"][b][k], 4), "z": z, "p": p}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
