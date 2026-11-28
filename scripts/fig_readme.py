"""Figures 9-12 for the README and EXPERIMENTS.md, drawn from the saved result files (no models are run).

fig9   token-identity control for the map (rotation_vs_ridge_results.json)
fig10  fact swap by hidden-state transfer (fact_swap_results.json)
fig11  loop-direction controls (loop_controls_results.json)
fig12  porting the loop trigger as a weight edit (lp_port_followup_results.json, lp_port_dense_controls.json)

Styling follows fig_loop_diagnostic.py: two validated colour slots, plus hatching and direct labels,
so no reading depends on colour alone.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE.parent / "docs" / "figures"
SURFACE, INK, INK2, GRID, NEUTRAL, LIGHT = "#ffffff", "#2b2a28", "#52514e", "#e4e3df", "#a3a29c", "#d6d5d0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
                     "grid.linewidth": 0.6, "pdf.fonttype": 42, "hatch.linewidth": 0.6, "legend.frameon": False,
                     "legend.fontsize": 7})


def load(name):
    return json.loads((HERE / name).read_text())


def style(ax, title):
    ax.set_title(title, loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def bars(ax, x, vals, width, color, hatch, fmt, dy):
    for xi, v in zip(x, vals):
        ax.bar(xi, v, width, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=1.0)
        ax.text(xi, v + dy, fmt.format(v), ha="center", fontsize=6, color=INK2)


# ---- fig9: how much of the map is the shared token --------------------------------------------
R = load("rotation_vs_ridge_results.json")
groups, rows = [], []
for p in R["pairs"]:
    short = {"Qwen/Qwen2.5-0.5B-Instruct": "Qwen-0.5B", "HuggingFaceTB/SmolLM2-135M": "Smol-135M"}[p["small"]]
    for lay in p["layers"]:
        groups.append(f"{short}\nlayer {lay['small_layer']}")
        rows.append(lay)
conds = [("trained_best", "full map from L's hidden state", BLUE, None),
         ("L_embedding", "map from L's input embedding only", ORANGE, "////"),
         ("random_best", "map from an untrained L", NEUTRAL, "xxxx")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
w = 0.26
for ax, kind, title in [(axes[0], "ridge", "(a) ridge map (any linear map)"),
                        (axes[1], "rotation", "(b) rotation map (orthogonal + scale)")]:
    for j, (key, lab, c, h) in enumerate(conds):
        x = [i + (j - 1) * w for i in range(len(rows))]
        bars(ax, x, [r[key][kind] for r in rows], w, c, h, "{:.2f}", 0.012)
        ax.bar(0, 0, color=c, hatch=h, edgecolor=SURFACE, label=lab)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    style(ax, title)
axes[0].set_ylabel("held-out R² (predicting S from Qwen-1.5B)")
axes[0].set_ylim(0, 0.8)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99))
save(fig, "fig9_token_identity")

# ---- fig10: fact swap -------------------------------------------------------------------------
F = load("fact_swap_results.json")["pairs"]
names = {"qwen1.5b": "Qwen-1.5B", "qwen0.5b": "Qwen-0.5B", "smol360m": "Smol-360M", "smol135m": "Smol-135M",
         "gemma270m": "Gemma-270M"}
pairs = list(F)
labels = [" to\n".join(names[m] for m in k.split("->")) for k in pairs]
conds = [("ridge", "ridge map of the donor's state", BLUE, None),
         ("rotation", "rotation map", ORANGE, "////"),
         ("subject_bag", "control: the subject's tokens only", NEUTRAL, "xxxx"),
         ("random_rotation", "control: random rotation", LIGHT, "...")]
fig, ax = plt.subplots(figsize=(7.0, 2.5))
w = 0.2
for j, (key, lab, c, h) in enumerate(conds):
    x = [i + (j - 1.5) * w for i in range(len(pairs))]
    vals = [F[k]["share_of_self"][key] * 100 for k in pairs]
    bars(ax, x, vals, w, c, h, "{:.0f}", 1.2)
    ax.bar(0, 0, color=c, hatch=h, edgecolor=SURFACE, label=lab)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(labels)
ax.set_ylabel("% of the own-state ceiling")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", ncol=2)
style(ax, "Fact swap: the recipient states the donor's fact (CounterFact, held-out facts)")
save(fig, "fig10_fact_swap")

# ---- fig11: loop-direction controls -------------------------------------------------------------
T = load("loop_controls_results.json")["test"]
conds = [("S_own", "S's own\ndirection", BLUE, None),
         ("L_same_map", "Qwen's, map fit\non the same texts", ORANGE, "////"),
         ("L_independent_map", "Qwen's, map fit\non other texts", ORANGE, None),
         ("L_embedding_only_independent_map", "embedding-only\nmap", NEUTRAL, "////"),
         ("random", "random\ndirection", NEUTRAL, "xxxx")]
fig, ax = plt.subplots(figsize=(5.2, 2.3))
for i, (k, lab, c, h) in enumerate(conds):
    v = T[k]["loop_rate"] * 100
    ax.bar(i, v, 0.62, color=c, hatch=h, edgecolor=SURFACE, linewidth=1.0)
    ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=6.5, color=INK2)
base = T["base"]["loop_rate"] * 100
ax.axhline(base, color=INK, lw=0.8, ls=(0, (4, 3)))
ax.text(-0.3, base + 0.6, f"no edit {base:.0f}%", ha="left", fontsize=6.5, color=INK)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c[1] for c in conds])
ax.set_ylabel("% of answers stuck in a loop")
style(ax, "Loop direction in SmolLM2-135M, strength 2, 200 held-out GSM8K prompts")
save(fig, "fig11_loop_controls")

# ---- fig12: porting the loop trigger as a weight edit ------------------------------------------
P = load("lp_port_followup_results.json")["test"]
C = load("lp_port_dense_controls.json")["test"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.28, width_ratios=[1.25, 1]))
ax = axes[0]
conds = [("S_own_trigger_8_neurons", "S's own trigger\n(8 neurons)", BLUE, None),
         ("ported_trigger_8_neurons", "ported trigger\n(8 neurons)", ORANGE, "////"),
         ("ported_trigger_all_neurons", "ported trigger\n(574 neurons)", ORANGE, None)]
w = 0.34
for i, (k, lab, c, h) in enumerate(conds):
    for j, (s, alpha) in enumerate([("2.0", 1.0), ("4.0", 0.55)]):
        v = P[f"{k} x{s}"]["loop_rate"] * 100
        ax.bar(i + (j - 0.5) * w, v, w, color=c, alpha=alpha, hatch=h, edgecolor=SURFACE, linewidth=1.0)
        ax.text(i + (j - 0.5) * w, v + 0.6, f"{v:.0f}", ha="center", fontsize=6, color=INK2)
        ax.text(i + (j - 0.5) * w, 0.8, f"×{s[0]}", ha="center", fontsize=6, color=INK, fontweight="bold")
base = P["base"]["loop_rate"] * 100
for a in axes:
    a.axhline(base, color=INK, lw=0.8, ls=(0, (4, 3)))
ax.text(2.6, base + 0.6, f"no edit {base:.0f}%", ha="right", fontsize=6.5, color=INK)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c[1] for c in conds])
ax.set_ylabel("% of answers stuck in a loop")
ax.set_ylim(0, 36)
style(ax, "(a) rank-1 edits, strength ×2 and ×4")
ax = axes[1]
conds = [("ported_all_neurons (real)", "ported\n(574)", ORANGE, None),
         ("port_program_on_shuffled_targets", "shuffled\ntargets", NEUTRAL, "////"),
         ("real_weights_shuffled_over_same_neurons", "weights\nshuffled", NEUTRAL, "xxxx"),
         ("real_weights_random_signs", "random\nsigns", NEUTRAL, "...")]
for i, (k, lab, c, h) in enumerate(conds):
    v = C[k]["loop_rate"] * 100
    ax.bar(i, v, 0.62, color=c, hatch=h, edgecolor=SURFACE, linewidth=1.0)
    ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.5, color=INK2)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c[1] for c in conds])
ax.set_ylim(0, 36)
style(ax, "(b) controls for the 574-neuron port, ×2")
save(fig, "fig12_weight_edit_port")
