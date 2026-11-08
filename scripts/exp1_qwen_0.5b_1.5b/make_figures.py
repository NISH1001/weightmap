"""Paper figures for the S<->L representation-map experiment.

Colour: validated categorical palette (dataviz reference instance), fixed slot order,
checked with validate_palette.js (CVD-safe adjacent pairs; slots 1-2 pass all-pairs).
Slots 3-5 sit below 3:1 contrast on white, so every figure that uses them also carries
hatching and direct labels (identity never by colour alone). Outputs PDF (vector) + PNG.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).parent
R = ROOT / "results"
FIG = ROOT.parents[1] / "docs" / "figures"   # repo-level docs/figures
FIG.mkdir(parents=True, exist_ok=True)

# ---- palette (light mode; print)
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"          # secondary text
GRID = "#e4e3df"          # hairline grid, one shade off surface
NEUTRAL = "#a3a29c"       # de-emphasised marks (controls)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 100..700

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.6, "legend.frameon": False,
})
COL1, COL2 = 3.4, 7.0   # single / double column widths (in)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def rd(name):
    return json.loads((R / name).read_text())


def ygrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True); ax.set_axisbelow(True)


# ============================================================ Fig 1: decomposition of the S/L gap
def fig_gap():
    rows = []
    for task, label in [("math", "MATH-500"), ("nq", "NQ-Open")]:
        S = rd(f"gen_S_{task}_test.json"); L = rd(f"gen_L_{task}_test.json")
        lat = rd(f"latent_{task}.json")
        n = len(S)
        both = sum(s["ok"] and l["ok"] for s, l in zip(S, L))
        blocked = sum(v > 0 for v in lat.values()); absent = len(lat) - blocked
        rows.append((label, n, both, blocked, absent))
    fig, ax = plt.subplots(figsize=(COL1, 1.35))
    for i, (label, n, both, blocked, absent) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, blocked, color=BLUE, height=0.52, edgecolor=SURFACE, linewidth=1.2)
        ax.barh(y, absent, left=blocked, color=ORANGE, height=0.52, edgecolor=SURFACE, linewidth=1.2, hatch="////")
        tot = blocked + absent
        ax.text(blocked / 2, y, f"{blocked}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        ax.text(blocked + absent / 2, y, f"{absent}", ha="center", va="center", color=INK, fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))
        ax.text(tot + 0.6, y, f"{blocked / tot:.0%} blocked", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("Problems solved by L (1.5B) but not S (0.5B), greedy")
    ax.set_xlim(0, 42); xgrid(ax)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    h = [mpl.patches.Patch(fc=BLUE, label="Blocked: S solves in ≥1 of 8 samples"),
         mpl.patches.Patch(fc=ORANGE, hatch="////", ec=SURFACE, label="Absent: S never solves")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, handlelength=1.4)
    save(fig, "fig1_gap_decomposition")


# ============================================================ Fig 2: CKA layer correspondence
def fig_cka():
    C = np.load(R / "cka.npy")
    match = {int(k): v for k, v in rd("layer_match.json").items()}
    cmap = LinearSegmentedColormap.from_list("blue_seq", SEQ)
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    im = ax.imshow(C.T, origin="lower", aspect="auto", cmap=cmap, vmin=max(0.0, C.min()), vmax=1.0,
                   interpolation="nearest")
    xs = sorted(match); ys = [match[a] for a in xs]
    ax.plot(xs, ys, color=SURFACE, lw=2.6, solid_capstyle="round")
    ax.plot(xs, ys, color=ORANGE, lw=1.4, marker="o", ms=3.2, mec=SURFACE, mew=0.6, label="best-matching L layer")
    for a in (10, 14):
        ax.annotate(f"S{a}$\\leftrightarrow$L{match[a]}", (a, match[a]), xytext=(a + 1.5, match[a] - 4.5), fontsize=6.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color=SURFACE, lw=0.8))
    ax.set_xlabel("S (0.5B) layer"); ax.set_ylabel("L (1.5B) layer")
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("linear CKA"); cb.outline.set_visible(False); cb.ax.tick_params(length=2)
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "fig2_cka_layers")


# ============================================================ Fig 3: map quality + direction agreement
def fig_map():
    ms = rd("map_stats.json")
    dirs = torch.load(R / "dirs.pt", weights_only=False)
    layers = sorted(int(k) for k in ms)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.1), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.35))
    cats = [("L->S", "L $\\rightarrow$ S"), ("S->L", "S $\\rightarrow$ L"), ("L->S shuffled-control", "L $\\rightarrow$ S\nshuffled"),
            ("S(l-1)->S(l) reference", "S($\\ell$$-$1) $\\rightarrow$ S($\\ell$)\nreference")]
    w = 0.36
    colors = [BLUE, ORANGE]; hatches = [None, "////"]
    for j, a in enumerate(layers):
        vals = [ms[str(a)][k] for k, _ in cats]
        x = np.arange(len(cats)) + (j - 0.5) * w
        a1.bar(x, vals, w, color=colors[j], hatch=hatches[j], edgecolor=SURFACE, linewidth=1.0,
               label=f"S layer {a} $\\leftrightarrow$ L layer {ms[str(a)]['L_layer']}")
        for xi, v in zip(x, vals):
            a1.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6, color=INK2)
    a1.axhline(0, color=INK2, lw=0.6)
    a1.set_xticks(range(len(cats))); a1.set_xticklabels([c for _, c in cats])
    a1.set_ylabel("held-out R² (closed-form ridge)"); a1.set_ylim(-0.25, 1.25); ygrid(a1)
    a1.tick_params(axis="x", length=0); a1.spines["bottom"].set_visible(False)
    a1.set_title("(a) Linear predictability between representations", loc="left")
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=6.5)

    tasks = [("math", "MATH"), ("nq", "NQ")]
    for j, a in enumerate(layers):
        vals = [dirs[t][a]["cos"] for t, _ in tasks]
        x = np.arange(len(tasks)) + (j - 0.5) * w
        a2.bar(x, vals, w, color=colors[j], hatch=hatches[j], edgecolor=SURFACE, linewidth=1.0)
        for xi, v in zip(x, vals):
            a2.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    a2.set_xticks(range(len(tasks))); a2.set_xticklabels([n for _, n in tasks])
    a2.set_ylim(0, 1.05); a2.set_ylabel(r"cos$(d_S,\ \mathrm{Map}(d_L))$"); ygrid(a2)
    a2.tick_params(axis="x", length=0); a2.spines["bottom"].set_visible(False)
    a2.set_title("(b) Correct-vs-wrong direction agreement", loc="left")
    save(fig, "fig3_map_quality")


# ============================================================ Fig 4: main result (forest plot)
METHODS = [("L->S", "L direction mapped into S"), ("transplant", "Transplant of L states"),
           ("S-native", "S's own direction"), ("random", "Random direction"), ("L (upper bound)", "L itself (ceiling)")]
KIND = {"L->S": "L-derived", "transplant": "L-derived", "S-native": "control", "random": "control",
        "L (upper bound)": "ceiling"}


def fig_main():
    rep = rd("s4_report.json")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0), sharey=True, gridspec_kw=dict(wspace=0.16))
    for ax, (task, title) in zip(axes, [("math", "MATH-500 (n = 110 held-out)"), ("nq", "NQ-Open (n = 300 held-out)")]):
        ci = rep[task]["delta_vs_base_all (mean, 95% CI)"]
        base = rep[task]["table"]["all"]["base"]
        for i, (m, _) in enumerate(METHODS):
            y = len(METHODS) - 1 - i
            mean, lo, hi = [100 * v for v in ci[m]]
            k = KIND[m]
            c = BLUE if k == "L-derived" else NEUTRAL if k == "control" else INK
            ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="round")
            ax.plot(mean, y, "o", ms=6 if k != "ceiling" else 5.5, color=c if k != "ceiling" else SURFACE,
                    mec=SURFACE if k != "ceiling" else INK, mew=1.0 if k != "ceiling" else 1.2, zorder=3)
            ax.text(1.0, y, f"{mean:+.1f}", va="center", ha="left", fontsize=6.5, color=INK2,
                    transform=ax.get_yaxis_transform())
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{title}, baseline {100 * base:.1f}%", loc="left")
        xgrid(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        lo_all = min(100 * ci[m][1] for m, _ in METHODS); hi_all = max(100 * ci[m][2] for m, _ in METHODS)
        pad = 0.06 * (hi_all - lo_all)
        ax.set_xlim(lo_all - pad, hi_all + pad)
    axes[0].set_yticks(range(len(METHODS))); axes[0].set_yticklabels([n for _, n in METHODS][::-1])
    h = [mpl.lines.Line2D([], [], color=BLUE, marker="o", lw=1.6, label="uses L (training-free)"),
         mpl.lines.Line2D([], [], color=NEUTRAL, marker="o", lw=1.6, label="control (no L information)"),
         mpl.lines.Line2D([], [], color=INK, marker="o", mfc=SURFACE, lw=1.6, label="L (1.5B) reference")]
    fig.supxlabel("Δ accuracy vs. unmodified S (percentage points, 95% paired-bootstrap CI)", fontsize=8, y=-0.04)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.55, 1.10), ncol=3)
    save(fig, "fig4_main_result")


# ============================================================ Fig 5: where do edits act? subgroup breakdown
def fig_breakdown():
    rep = rd("s4_report.json")
    conds = [("base", "Unmodified S", BLUE, None), ("L->S", "L direction $\\rightarrow$ S", ORANGE, "////"),
             ("transplant", "Transplant", AQUA, "...."), ("S-native", "S's own direction", YELLOW, "\\\\\\\\"),
             ("random", "Random", MAGENTA, "xxxx")]
    groups = [("target-blocked (L ok, S wrong, S pass@8>0)", "Blocked"),
              ("target-absent (L ok, S wrong, S pass@8=0)", "Absent"),
              ("S-already-correct (damage check)", "S already\ncorrect")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    w = 0.16
    for ax, (task, title) in zip(axes, [("math", "MATH-500"), ("nq", "NQ-Open")]):
        tab = rep[task]["table"]
        for j, (k, name, c, h) in enumerate(conds):
            x = np.arange(len(groups)) + (j - 2) * w
            vals = [100 * tab[g][k] for g, _ in groups]
            ax.bar(x, vals, w, color=c, hatch=h, edgecolor=SURFACE, linewidth=0.8, label=name)
            for xi, v in zip(x, vals):
                if v == 0:   # make zeros visible: a short baseline tick in the series colour + "0"
                    ax.plot([xi - w * 0.4, xi + w * 0.4], [0.6, 0.6], color=c, lw=2.2, solid_capstyle="butt")
                    ax.text(xi, 2.0, "0", ha="center", va="bottom", fontsize=5.5, color=INK2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{lab}\n(n = {tab[g]['n']})" for g, lab in groups])
        ax.set_title(title, loc="left"); ygrid(ax); ax.set_ylim(0, 105)
        ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("accuracy on held-out subgroup (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=5)
    save(fig, "fig5_subgroup_breakdown")


if __name__ == "__main__":
    fig_gap(); fig_cka(); fig_map(); fig_main(); fig_breakdown()
