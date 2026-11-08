"""Stage 1: greedy outputs of S and L on every split -> partition problems.
target = L correct & S wrong (the Pl \\ Ps set)."""
from common import *

SPLITS = [(t, sp, None) for (t, sp) in N]

for name, tag in [(S_NAME, "S"), (L_NAME, "L")]:
    tok, model = load(name)
    for task, split, n in SPLITS:
        f = f"gen_{tag}_{task}_{split}.json"
        if (OUT / f).exists():
            continue
        data = split_data(task, split)
        log(tag, task, split, len(data))
        ps = [prompt_text(tok, task, d["q"]) for d in data]
        outs = generate(tok, model, ps, max_new=max_new(task), bs=64 if task == "math" else 200)
        rows = [{"id": d["id"], "out": o[0], "ok": correct(task, o[0], d["gold"])} for d, o in zip(data, outs)]
        save(f, rows)
        log(f"  acc={sum(r['ok'] for r in rows) / len(rows):.3f}")
    del model
    torch.mps.empty_cache()

for task, split, _ in SPLITS:
    S, L = read(f"gen_S_{task}_{split}.json"), read(f"gen_L_{task}_{split}.json")
    cats = {"both": 0, "target(L only)": 0, "S only": 0, "neither": 0}
    for s, l in zip(S, L):
        k = "both" if s["ok"] and l["ok"] else "target(L only)" if l["ok"] else "S only" if s["ok"] else "neither"
        cats[k] += 1
    log(task, split, cats)
