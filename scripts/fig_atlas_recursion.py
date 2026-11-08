"""Figures for the atlas + recursion run (palette and styling as scripts/exp1_qwen_0.5b_1.5b/make_figures.py:
validated categorical slots 1-2, hatching as secondary encoding, direct labels)."""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE.parent / "docs" / "figures"
R = json.loads((HERE / "atlas_recursion_results.json").read_text())
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "legend.frameon": False, "hatch.linewidth": 0.6})
SHORT = {"Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M",
         "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M", "google/gemma-3-270m": "Gemma-3-270M",
         "EleutherAI/pythia-160m": "Pythia-160M"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


# ---------------- Fig 6: atlas
at = R["atlas"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharex=True, gridspec_kw=dict(wspace=0.3))
for ax, key, title, c, mk in [(axes[0], "mean_r2_S_to_L", "(a) small $\\rightarrow$ Qwen-1.5B: how much of L\nthe small model captures", BLUE, "o"),
                              (axes[1], "mean_r2_L_to_S", "(b) Qwen-1.5B $\\rightarrow$ small: how much of the\nsmall model L contains", ORANGE, "s")]:
    x = [e["bits_per_byte"] for e in at]; y = [e[key] for e in at]
    ax.scatter(x, y, s=34, color=c, marker=mk, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for e, xi, yi in zip(at, x, y):
        ax.annotate(SHORT[e["small"]], (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.set_xlabel("small-model bits per byte  ($\\leftarrow$ more capable)")
    ax.set_ylabel("held-out R² (text holdout)")
    ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_ylim(0.4, 0.75)
    ax.set_xlim(0.1, 1.15)
save(fig, "fig6_atlas")

# ---------------- Fig 7: recursion
rec = R["recursion"]
pairs = [("Qwen2.5-1.5B-Instruct -> SmolLM2-360M", "1.5B $\\rightarrow$ 360M\n(via 0.5B)"),
         ("Qwen2.5-0.5B-Instruct -> SmolLM2-135M", "0.5B $\\rightarrow$ 135M\n(via 360M)"),
         ("Qwen2.5-1.5B-Instruct -> SmolLM2-135M", "1.5B $\\rightarrow$ 135M\n(via 0.5B, 360M)")]
steps = [v for k, v in rec.items() if k.startswith("step")]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
w = 0.36
xs = np.arange(len(pairs))
d = [rec[k]["direct"] for k, _ in pairs]; c = [rec[k]["composed_via_chain"] for k, _ in pairs]
ax.bar(xs - w / 2, d, w, color=BLUE, edgecolor=SURFACE, linewidth=1.0, label="direct map")
ax.bar(xs + w / 2, c, w, color=ORANGE, hatch="////", edgecolor=SURFACE, linewidth=1.0, label="composed step maps")
for i in range(len(pairs)):
    ax.text(xs[i] - w / 2, d[i] + 0.01, f"{d[i]:.2f}", ha="center", fontsize=6, color=INK2)
    ax.text(xs[i] + w / 2, c[i] + 0.01, f"{c[i]:.2f}", ha="center", fontsize=6, color=INK2)
# naive expectation if step errors compounded: product of step R^2
prod = {0: steps[0] * steps[1], 1: steps[1] * steps[2], 2: steps[0] * steps[1] * steps[2]}
for i in range(len(pairs)):
    ax.plot([xs[i] - 0.4, xs[i] + 0.4], [prod[i]] * 2, color=INK, lw=1.0, ls=(0, (1, 1.5)))
ax.plot([], [], color=INK, lw=1.0, ls=(0, (1, 1.5)), label="product of step R² (if errors compounded)")
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in pairs])
ax.set_ylabel("held-out R²"); ax.set_ylim(0, 0.85)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1)
save(fig, "fig7_recursion")
