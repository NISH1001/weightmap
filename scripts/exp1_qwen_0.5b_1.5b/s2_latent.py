"""Stage 2: for test-set target problems (L right, S greedy wrong), sample S k=16 times.
blocked = S solves it at least once (ability latent); absent = never."""
from common import *

K = 8
tok, model = load(S_NAME)
for task in ["math", "nq"]:
    f = f"latent_{task}.json"
    if (OUT / f).exists():
        continue
    data = split_data(task, "test")
    S, L = read(f"gen_S_{task}_test.json"), read(f"gen_L_{task}_test.json")
    tgt = [d for d, s, l in zip(data, S, L) if l["ok"] and not s["ok"]]
    log(task, "target", len(tgt))
    ps = [prompt_text(tok, task, d["q"]) for d in tgt]
    outs = generate(tok, model, ps, max_new=max_new(task), sample=True, n=K, bs=8 if task == "math" else 25)
    res = {}
    for d, os_ in zip(tgt, outs):
        res[d["id"]] = sum(correct(task, o, d["gold"]) for o in os_)
    save(f, res)
    nb = sum(v > 0 for v in res.values())
    log(f"  blocked={nb} absent={len(res) - nb}  (pass@{K} on target = {nb / len(res):.3f})")
