#!/usr/bin/env python3
"""Draw the Hybrid + V8 architecture figure for the capstone report."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "architecture_hybrid_v8.png"

NAVY = "#0B3D5C"
STEEL = "#1F4E79"
TEAL = "#0E7C7B"
AMBER = "#C47B17"
ROSE = "#9B3A4A"
SLATE = "#5A6A75"
LINE = "#90A4AE"
PAPER = "#F7F5F0"
CARD = "#FFFFFF"
ALT = "#E8EEF4"
LEARN = "#E8F3F1"
PHYS = "#F3EEE4"
SCORE = "#EDE6F4"


def box(ax, x, y, w, h, text, *, fc=CARD, ec=NAVY, tc=NAVY, fs=8.2, lw=1.15, radius=0.08, weight="medium"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, color=tc, fontweight=weight, zorder=3,
        linespacing=1.25, wrap=False,
    )
    return p


def arrow(ax, x1, y1, x2, y2, color=STEEL, lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=11, linewidth=lw,
            color=color, zorder=1, shrinkA=0, shrinkB=0,
        )
    )


def label(ax, x, y, text, *, fs=7.4, color=SLATE, ha="center", va="center", style="italic"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, style=style, zorder=4)


def band(ax, x, y, w, h, title, fc):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none", zorder=0))
    ax.text(x + 0.12, y + h - 0.16, title, fontsize=8.4, color=NAVY, fontweight="bold", va="top", zorder=1)


def draw():
    fig = plt.figure(figsize=(13.4, 9.6), dpi=180, facecolor=PAPER)
    ax = fig.add_axes([0.025, 0.03, 0.95, 0.93])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14.4)
    ax.axis("off")
    ax.set_facecolor(PAPER)

    ax.text(
        10, 14.15,
        "How the model works  ·  Hybrid pipeline + V8 associator",
        ha="center", va="center", fontsize=13.5, color=NAVY, fontweight="bold",
    )
    ax.text(
        10, 13.72,
        "The net scores gated pairs.  Kalman owns state.  Hungarian owns uniqueness.",
        ha="center", va="center", fontsize=8.6, color=SLATE, style="italic",
    )

    # ----- Panel A: pipeline -----
    ax.add_patch(FancyBboxPatch((0.15, 7.55), 19.7, 5.95, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FBFBFA", edgecolor=LINE, linewidth=0.8, zorder=0))
    ax.text(0.4, 13.22, "A   Hybrid correlator  —  one process, two learned scoring slots",
            fontsize=9.6, color=NAVY, fontweight="bold", va="center")

    # inputs
    box(ax, 0.4, 11.55, 2.35, 1.15, "Multi-radar\nplots\nPSR + SSR", fc=ALT, fs=7.8)
    box(ax, 0.4, 10.15, 2.35, 1.15, "Live tracks\nKF state", fc=ALT, fs=7.8)

    box(ax, 3.05, 10.75, 2.45, 1.55, "Clutter MLP\nunary reject\nPSR false alarms", fc=LEARN, ec=TEAL, fs=7.6)
    box(ax, 5.85, 10.75, 2.7, 1.55, "2 km cluster\nconnected\ncomponents", fc=PHYS, ec=AMBER, fs=7.6)
    box(ax, 8.9, 10.75, 2.55, 1.55, "Project track\nto meas_t\ndt = tₘ − tₜ", fc=PHYS, ec=AMBER, fs=7.6)
    box(ax, 11.8, 10.75, 2.55, 1.55, "8 km assign\nHungarian\ncost = 1 − p", fc=PHYS, ec=AMBER, fs=7.6)
    box(ax, 14.7, 10.75, 2.45, 1.55, "Async CV KF\nupdate at tₘ\nno time-drag", fc=PHYS, ec=AMBER, fs=7.6)
    box(ax, 17.45, 10.75, 2.15, 1.55, "M/N manager\n3 hits / ~10\ncoasts", fc=PHYS, ec=AMBER, fs=7.6)

    arrow(ax, 2.75, 12.1, 3.05, 11.7)
    arrow(ax, 2.75, 10.7, 3.05, 11.2)
    for x1, x2 in [(5.5, 5.85), (8.55, 8.9), (11.45, 11.8), (14.35, 14.7), (17.15, 17.45)]:
        arrow(ax, x1, 11.52, x2, 11.52)

    # scorer bank
    ax.add_patch(FancyBboxPatch((5.85, 7.85), 8.7, 2.55, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=SCORE, edgecolor="#6B4C8A", linewidth=1.1, zorder=1))
    ax.text(10.2, 10.15, "Learned scorer  (only this block is a neural net)",
            ha="center", fontsize=8.0, color="#4A2F66", fontweight="bold")

    box(ax, 6.05, 8.1, 2.5, 1.7, "MLP pair\n4–6 features\nthis pair only", fc="#F7F2FC", ec="#6B4C8A", fs=7.5)
    box(ax, 8.7, 8.1, 2.7, 1.7, "V8 transformer\nset self-attn\nthen pair head", fc="#F7F2FC", ec="#6B4C8A", fs=7.5)
    box(ax, 11.55, 8.1, 2.8, 1.7, "Compose\nensemble ½+½\nor split MLP/V8", fc="#F7F2FC", ec="#6B4C8A", fs=7.5)

    # drop lines from scorer to cluster + assign
    ax.plot([7.2, 7.2], [9.8, 10.75], color="#6B4C8A", lw=1.15, zorder=1)
    ax.plot([10.05, 13.05], [9.8, 10.75], color="#6B4C8A", lw=1.15, zorder=1)
    label(ax, 6.35, 10.42, "p > 0.5 → cluster edge", fs=6.5, color="#6B4C8A", ha="center")
    label(ax, 13.55, 10.42, "p → cost = 1−p", fs=6.5, color="#6B4C8A", ha="left")

    ax.text(17.4, 9.15, "Not learned\nKalman · gates\nHungarian · M/N", ha="center", va="center",
            fontsize=7.4, color=SLATE, style="italic")

    # ----- Panel B: V8 internals -----
    ax.add_patch(FancyBboxPatch((0.15, 0.2), 19.7, 7.15, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FBFBFA", edgecolor=LINE, linewidth=0.8, zorder=0))
    ax.text(0.4, 7.05, "B   AssociationTransformerV8  —  SuperGlue-style pair scorer, not a tracker",
            fontsize=9.6, color=NAVY, fontweight="bold", va="center")

    # tokens
    box(ax, 0.4, 5.15, 3.15, 1.55, "Track tokens\nprojected state\nrole = track", fc=ALT, fs=7.7)
    box(ax, 0.4, 3.35, 3.15, 1.55, "Plot / meta tokens\nrole = meas\nPSR or SSR", fc=ALT, fs=7.7)

    box(ax, 3.85, 3.35, 3.55, 3.35,
        "Token build\n15-d numeric  →  Linear 64\n+ 5 embeds (64-d each)\nrole · type · sensor\nMode-3A · Mode-S hash",
        fc=LEARN, ec=TEAL, fs=7.5)

    arrow(ax, 3.55, 5.9, 3.85, 5.4)
    arrow(ax, 3.55, 4.1, 3.85, 4.5)

    # two self-attn
    box(ax, 7.7, 5.15, 3.35, 1.55, "Self-attn  tracks\n2 layers · 4 heads\nd=64  FFN 256  GELU", fc=SCORE, ec="#6B4C8A", fs=7.5)
    box(ax, 7.7, 3.35, 3.35, 1.55, "Self-attn  plots\nsame weights\nno cross-attention", fc=SCORE, ec="#6B4C8A", fs=7.5)
    arrow(ax, 7.4, 5.05, 7.7, 5.85)
    arrow(ax, 7.4, 4.95, 7.7, 4.15)

    label(ax, 9.35, 4.95, "context stays on its own side", fs=6.6, color="#6B4C8A")

    # rel + heads
    box(ax, 11.35, 3.35, 3.55, 3.35,
        "rel_ij  (12-d, explicit)\ndx dy dz dist  Δ|v| cos_v\nΔaz Δel  dt\nMode-3A/S match  same radar",
        fc=PHYS, ec=AMBER, fs=7.4)
    arrow(ax, 11.05, 5.9, 11.35, 5.4)
    arrow(ax, 11.05, 4.1, 11.35, 4.5)

    box(ax, 15.2, 5.25, 4.4, 1.45, "score_pairs\n[hᵢ ; hⱼ ; rel] → MLP → logit\n2 km cluster edges", fc=LEARN, ec=TEAL, fs=7.5)
    box(ax, 15.2, 3.35, 4.4, 1.65, "score_assignment\nS ∈ ℝᵀˣᴹ  +  dustbin ∈ ℝᵀ\n8 km Hungarian costs", fc=LEARN, ec=TEAL, fs=7.5)
    arrow(ax, 14.9, 5.05, 15.2, 5.9)
    arrow(ax, 14.9, 4.95, 15.2, 4.2)

    # forbidden / allowed strip
    box(ax, 0.4, 0.45, 9.5, 2.6,
        "Allowed     p = σ(logit)     cluster if p > 0.5     cost = 1 − p\n"
        "Label       track_id equality — never an input feature\n"
        "Identity    Mode-3A / Mode-S only; pad 0 + has_mode_3a flag",
        fc="#EEF5F1", ec=TEAL, fs=7.4, weight="normal")
    box(ax, 10.15, 0.45, 9.45, 2.6,
        "Forbidden   residual Δs    existence / init heads    GRU memory\n"
        "            scoring outside the 2 km / 8 km gates\n"
        "That is V7.  V8 does not own birth, death, or time.",
        fc="#F8EEEE", ec=ROSE, fs=7.4, weight="normal")

    fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    draw()
