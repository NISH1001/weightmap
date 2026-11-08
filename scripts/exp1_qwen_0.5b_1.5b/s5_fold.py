"""Stage 5: fold the best static L->S steering vector into S's weights (a bias on one MLP down-projection),
verify it is exactly the same model as the hook, and save the edited checkpoint. No training."""
import torch.nn as nn
from common import *
import s4_steer as s4  # reuses loaded models/dirs; s4 must already have finished (all runs cached)

rep = read("s4_report.json")
for task in ["math", "nq"]:
    _, a, c = rep[task]["best_configs"]["L->S"]
    v = s4.vec_for(task, "L->S", a, c)
    tok, S = s4.tokS, s4.S
    enc = tok([prompt_text(tok, task, "What is 17 * 23?")], return_tensors="pt").to(DEV)
    with torch.no_grad(), add_vec(S, a, v):
        ref = S(**enc).logits.float()
    mlp = S.model.layers[a].mlp
    old = mlp.down_proj
    new = nn.Linear(old.in_features, old.out_features, bias=True).to(DEV, DTYPE)
    new.weight.data.copy_(old.weight.data); new.bias.data.copy_(v.to(DTYPE))
    mlp.down_proj = new
    with torch.no_grad():
        got = S(**enc).logits.float()
    diff = (ref - got).abs().max().item()
    log(f"{task}: layer {a} bias edit, {v.numel()} params changed "
        f"({v.numel() / sum(p.numel() for p in S.parameters()) * 100:.5f}% of S); max |logit diff| vs hook = {diff:.4g}")
    S.save_pretrained(OUT / f"S_edited_{task}"); tok.save_pretrained(OUT / f"S_edited_{task}")
    mlp.down_proj = old
