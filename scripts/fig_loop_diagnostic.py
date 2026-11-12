"""Figure for the loop-diagnostic run (styling as the other figures; slots 1-2 validated, hatching, direct labels)."""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE.parent / "docs" / "figures"
R = json.loads((HERE / "loop_diagnostic_fixed_18x2.json").read_text())["test"]
SURFACE, INK2, GRID, NEUTRAL = "#ffffff", "#52514e", "#e4e3df", "#a3a29c"
BLUE, ORANGE = "#2a78d6", "#eb6834"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
                     "grid.linewidth": 0.6, "pdf.fonttype": 42, "hatch.linewidth": 0.6})
conds = [("base", "no steering", NEUTRAL, None), ("S-native", "S's own\ndirection", BLUE, None),
         ("L->S", "Qwen's direction\nmapped into S", ORANGE, "////"), ("random", "random\ndirection", NEUTRAL, "xxxx")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.3))
for ax, key, title, fmt, scale in [(axes[0], "loop_rate", "(a) loop rate on 200 held-out GSM8K prompts", "{:.1%}", 100),
                                   (axes[1], "early_nll_qwen", "(b) early coherence: Qwen NLL/token (lower = better)", "{:.2f}", 1)]:
    for i, (k, lab, c, h) in enumerate(conds):
        v = R[k][key]
        ax.bar(i, v * scale, 0.62, color=c, hatch=h, edgecolor=SURFACE, linewidth=1.0)
        ax.text(i, v * scale * 1.02 + (0.4 if scale == 100 else 0.02), fmt.format(v), ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds])
    ax.set_title(title, loc="left"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)
axes[0].set_ylabel("% of answers stuck in a loop"); axes[1].set_ylabel("NLL / token")
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig8_loop_diagnostic.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("wrote fig8_loop_diagnostic")
