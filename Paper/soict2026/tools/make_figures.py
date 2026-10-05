"""Publication figures for the SOICT 2026 CAD-VR manuscript.

Inputs are the committed benchmark snapshot (benchmarks/agent/results/
soict-final-tol1) and data/paper_stats.json from derive_stats.py. No model or
network calls. Run from the repository root with an environment that has
matplotlib (for example the root .venv):

    .venv/bin/python Paper/soict2026/tools/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
SNAP = REPO / "benchmarks/agent/results/soict-final-tol1/test"
FIG = PAPER / "figures"
STATS = json.loads((PAPER / "data/paper_stats.json").read_text())

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
    "gray_dark": "#4D4D4D", "gray_mid": "#767676",
}
# Semantic roles: blues = CAD-VR (proposed), reds = agent baselines, grays = no agents.
VARIANT_STYLE = {
    "A": dict(color=PALETTE["gray_mid"], marker="s", label="A  Retrieval"),
    "D": dict(color=PALETTE["neutral"], marker="D", label="D  Jev rerank"),
    "B": dict(color=PALETTE["red_2"], marker="^", label="B  Always Codex"),
    "C": dict(color=PALETTE["red_strong"], marker="v", label="C  Always both"),
    "E": dict(color=PALETTE["blue_main"], marker="o", label="E  CAD-VR (holistic)"),
    "F": dict(color=PALETTE["teal"], marker="P", label="F  CAD-VR (constraint)"),
}
TEXTWIDTH_IN = 4.8  # LNCS text block, 122 mm
SCALE = 2.0         # draw at 2x, include at \textwidth -> 7.5-8 pt text


def setup():
    for path in font_manager.findSystemFonts():
        if "texgyreheros" in path.lower():
            font_manager.fontManager.addfont(path)
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in ("TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans")
                if name in installed]
    plt.rcParams.update({
        "font.family": families + ["sans-serif"],
        "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
        "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 13,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2.0,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6, "xtick.major.size": 5, "ytick.major.size": 5,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Heros", "mathtext.it": "TeX Gyre Heros:italic",
    })


def finalize(fig, name, pad=0.6):
    fig.tight_layout(pad=pad)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", f"figures/{name}.pdf")


def panel_label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", ha="right")


def fig_tradeoff():
    """Strict R@1 against latency and System-2 invocations (TEST, N=86)."""
    intervals = json.loads((SNAP / "main/paper_intervals.json").read_text())["full"]["summary"]
    main = STATS["main"]
    replay = STATS["router_replay"]["E"]["accept"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN * SCALE, 3.15 * SCALE * .57))
    xs = {
        "latency": {v: main[v]["latency_p50_s"] for v in main},
        "agents": {v: main[v]["codex_calls"] + main[v]["claude_calls"] for v in main},
    }
    nudges = {"latency": {"A": (-9, 6), "D": (9, -16), "B": (0, -20), "C": (15, 3), "E": (-10, 9), "F": (6, -19)},
              "agents": {"A": (8, 5), "D": (8, -16), "B": (8, -12), "C": (-17, -12), "E": (-16, 9), "F": (15, 9)}}
    for ax, key, xlabel in ((axes[0], "latency", "Median end-to-end latency (s)"),
                            (axes[1], "agents", "System-2 agent invocations / query")):
        for v, style in VARIANT_STYLE.items():
            r = intervals[v]["r1"]
            x, y = xs[key][v], 100 * r["mean"]
            lo, hi = 100 * r["ci95"][0], 100 * r["ci95"][1]
            if key == "latency":  # CIs once; panel (b) shares the y values
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=style["color"], elinewidth=1.6,
                            capsize=4, alpha=.55, zorder=2)
            ax.scatter(x, y, s=210 if v == "C" else 150, color=style["color"], marker=style["marker"], edgecolor="black",
                       linewidth=1.1, zorder=4)
            dx, dy = nudges[key][v]
            ax.annotate(v, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=14,
                        fontweight="bold", color=style["color"] if v not in "AD" else PALETTE["gray_dark"],
                        ha="center", va="center")
        # Post-hoc counterfactual: accept the router's STOP proposal (no deployment claim).
        x2 = replay["latency_p50"] if key == "latency" else replay["agents"]
        y2 = 100 * replay["r1"]
        # Do not imply an experimentally measured operating curve between a
        # deployed run and an offline truncation of that run.
        ax.scatter(x2, y2, s=150, facecolor="white", edgecolor=PALETTE["blue_main"], linewidth=2.0,
                   marker="o", zorder=5)
        ax.annotate(r"E$^{\dagger}$", (x2, y2), xytext=(0, -19), textcoords="offset points", fontsize=14,
                    fontweight="bold", color=PALETTE["blue_main"], ha="center")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 75)
        ax.set_yticks([0, 20, 40, 60])
        ax.grid(axis="y", color="#E6E6E6", lw=1, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Strict R@1 (%)")
    axes[0].set_xlim(-4, 95)
    axes[1].set_xlim(-0.15, 2.25)
    axes[1].set_xticks([0, 0.5, 1, 1.5, 2])
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    handles = [Line2D([], [], ls="", marker=s["marker"], markersize=10, markerfacecolor=s["color"],
                      markeredgecolor="black", label=s["label"]) for s in VARIANT_STYLE.values()]
    handles.append(Line2D([], [], ls="", marker="o", markersize=10, markerfacecolor="white",
                          markeredgecolor=PALETTE["blue_main"], markeredgewidth=2,
                          label=r"E$^{\dagger}$  E, router STOP accepted (replay)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.1, handletextpad=0.3, fontsize=12.5)
    finalize(fig, "tradeoff")


def fig_grouping():
    """Frame-strict vs group-level vs video-level success (TEST, N=86)."""
    summary = json.loads((SNAP / "main/video-report/summary.json").read_text())["levels"]["full"]["paired"]["summary"]
    variants = ["A", "B", "C", "E", "F"]
    metrics = [("strict", "Frame strict", PALETTE["red_2"], ""),
               ("group_moment", "Group moment", PALETTE["blue_secondary"], ""),
               ("video", "Video", PALETTE["green_3"], "//")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN * SCALE, 2.05 * SCALE * .8), sharey=True)
    width = 0.26
    x = np.arange(len(variants))
    for ax, k, title in ((axes[0], "1", "Recall@1"), (axes[1], "5", "Recall@5")):
        for i, (m, label, color, hatch) in enumerate(metrics):
            vals = [100 * summary[v][f"{m}_r{k}"] for v in variants]
            bars = ax.bar(x + (i - 1) * width, vals, width, color=color, edgecolor="black", linewidth=1.1,
                          hatch=hatch, label=label, zorder=3)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.0f}", ha="center", va="bottom",
                        fontsize=10.5, color=PALETTE["gray_dark"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}" for v in variants], fontweight="bold")
        for tick, v in zip(ax.get_xticklabels(), variants):
            tick.set_color(VARIANT_STYLE[v]["color"] if v not in "A" else PALETTE["gray_dark"])
        ax.set_title(title, pad=6)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#E6E6E6", lw=1, zorder=0)
    axes[0].set_ylabel("Success (%)")
    axes[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.0), fontsize=11.5,
                   handlelength=1.4, columnspacing=0.9)
    finalize(fig, "grouping")


def fig_cumulative():
    """Independent cumulative-hint levels on the fixed 24-query TEST cohort."""
    levels = json.loads((SNAP / "cumulative/paper_intervals.json").read_text())["hint_levels"]
    import csv
    with open(SNAP / "cumulative/hint_results.csv") as f:
        rows = list(csv.DictReader(f))
    order = [("h1", "H1"), ("h2", "H1+H2"), ("full", "Full")]
    variants = ["A", "C", "F"]

    def mean_of(stage, v, key):
        vals = [float(r[key] or 0) for r in rows if r["hint_stage"] == stage and r["variant"] == v]
        return float(np.mean(vals)) if vals else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN * SCALE, 2.0 * SCALE * .82))
    x = np.arange(len(order))
    for ax, metric, ylabel in ((axes[0], "video_r1", "Video R@1 (%)"), (axes[1], "r1", "Strict R@1 (%)")):
        for v in variants:
            st = VARIANT_STYLE[v]
            ys, lo, hi = [], [], []
            for stage, _ in order:
                s = levels[stage]["summary"][v][metric]
                ys.append(100 * s["mean"]); lo.append(100 * s["ci95"][0]); hi.append(100 * s["ci95"][1])
            ax.plot(x, ys, color=st["color"], marker=st["marker"], markersize=9, lw=2.4,
                    markeredgecolor="black", markeredgewidth=0.9, ls="--" if v == "A" else "-",
                    label=st["label"])
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in order])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#E6E6E6", lw=1)
    ax = axes[2]
    for v in ("C", "F"):
        st = VARIANT_STYLE[v]
        ax.plot(x, [mean_of(s, v, "codex_calls") + mean_of(s, v, "claude_calls") for s, _ in order],
                color=st["color"], marker=st["marker"], markersize=9, lw=2.4, markeredgecolor="black",
                markeredgewidth=0.9, label=v)
    ax.set_ylim(1.5, 2.1)
    ax.set_ylabel("Agents / query")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in order])
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.bar(x, [mean_of(s, "F", "jev_calls") for s, _ in order], 0.42, color=PALETTE["green_2"],
             edgecolor="black", linewidth=1.0, alpha=0.55, zorder=0)
    twin.set_ylim(0, 30)
    twin.set_ylabel("Jev calls / query (F, bars)", color=PALETTE["gray_dark"])
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)
    for a, lab in zip(axes, ("(a)", "(b)", "(c)")):
        panel_label(a, lab)
    axes[0].legend(loc="lower right", fontsize=11.5, handlelength=1.8)
    finalize(fig, "cumulative")


def fig_controller():
    """Time scales, router stopping signal and verifier reliability (TEST, variant E)."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN * SCALE, 2.05 * SCALE * .84),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    # (a) per-layer wall time, log scale
    ax = axes[0]
    L = STATS["latency_samples"]
    groups = [("Jev call", L["jev_call_s"], PALETTE["green_3"]),
              ("Retrieval", L["retrieval_s"], PALETTE["gray_mid"]),
              ("Claude", L["claude_s"], PALETTE["red_2"]),
              ("Codex", L["codex_s"], PALETTE["red_strong"]),
              ("E total", L["end_to_end_s"], PALETTE["blue_main"])]
    bp = ax.boxplot([g[1] for g in groups], orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.8), whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3))
    for patch, (_, _, color) in zip(bp["boxes"], groups):
        patch.set_facecolor(color); patch.set_edgecolor("black"); patch.set_linewidth(1.1)
    rng = np.random.default_rng(2026)
    for i, (_, vals, color) in enumerate(groups, 1):
        vals = np.asarray(vals)
        ax.scatter(vals, i + rng.uniform(-0.17, 0.17, len(vals)), s=4, color="black", alpha=0.18, zorder=3)
        ax.text(vals.max() * 1.3, i, f"{np.median(vals):.1f} s" if np.median(vals) >= 1 else f"{np.median(vals):.2f} s",
                ha="left", va="center", fontsize=10.5)
    ax.set_xscale("log")
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlabel("Wall time (s, log)")
    ax.set_xlim(0.15, 3000)
    # (b) router P(STOP) after the first agent, split by current top-1 correctness
    ax = axes[1]
    d = STATS["post_agent_decisions"]["E"]
    bins = np.linspace(0, 1, 11)
    ok = [x["p_stop"] for x in d if x["top1_correct"]]
    bad = [x["p_stop"] for x in d if not x["top1_correct"]]
    ax.hist([ok, bad], bins=bins, stacked=True, color=[PALETTE["blue_secondary"], PALETTE["red_1"]],
            edgecolor="black", linewidth=1.0, label=["top-1 correct", "top-1 wrong"])
    ax.axvline(0.5, color=PALETTE["gray_dark"], lw=1.4, ls="--")
    ax.set_xlabel(r"Router $P(\mathrm{STOP})$ after agent 1")
    ax.set_ylabel("Queries")
    ax.legend(loc="upper left", fontsize=10.5, handlelength=1.1, borderaxespad=0.2)
    ax.set_xlim(0, 1)
    hi_ = [x for x in d if x["p_stop"] > .5]
    ax.text(0.03, 0.66, f"$P>0.5$: {len(hi_)} STOP\nproposals ({sum(x['top1_correct'] for x in hi_)} correct),\nall overridden",
            transform=ax.transAxes, fontsize=10, va="top", color=PALETTE["gray_dark"], zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))
    # (c) reliability of verifier probabilities
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=PALETTE["gray_mid"], lw=1.3, ls=":")
    for v, color, marker in (("E", PALETTE["blue_main"], "o"), ("F", PALETTE["teal"], "P")):
        rel = STATS["reliability"][v]
        pts = [(b["conf"], b["acc"], b["n"]) for b in rel["bins"] if b["n"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.8, alpha=0.8)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=[24 + 2.2 * np.sqrt(p[2]) * 6 for p in pts],
                   color=color, marker=marker, edgecolor="black", linewidth=0.9, zorder=3,
                   label=f"{v}: ECE {rel['ece']:.3f}")
    ax.set_xlabel("Verifier probability")
    ax.set_ylabel("Strict correctness")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=11, handletextpad=0.2)
    for a, lab in zip(axes, ("(a)", "(b)", "(c)")):
        panel_label(a, lab)
    finalize(fig, "controller")


if __name__ == "__main__":
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_tradeoff()
    fig_grouping()
    fig_cumulative()
    fig_controller()
