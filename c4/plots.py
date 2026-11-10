"""Charts for the README: training progress and win-confidence vs. ground truth.

  python -m c4.plots --run runs/main --results results --out docs/img
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
WIN, DRAW, LOSS = "#1baf7a", "#a3a29c", "#e34948"


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)


def training(run: str, out: str):
    rows = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    g = [r["gen"] for r in rows]
    games = [sum(r["games"] for r in rows[: i + 1]) for i in range(len(rows))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURF)
    ax = axes[0]
    _style(ax, "Move accuracy on 1,000 solver-labelled positions")
    ax.plot(g, [100 * r["policy_acc"] for r in rows], color=S1, lw=2, label="network alone (no search)")
    sg = [(r["gen"], 100 * r["search_acc"]) for r in rows if "search_acc" in r]
    if sg:
        ax.plot(*zip(*sg), color=S2, lw=2, marker="o", ms=5, label="network + 200-sim search")
    ax.set_xlabel("generation"); ax.set_ylabel("% of positions where the move is optimal")
    ax.set_ylim(min(30, ax.get_ylim()[0]), 100)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    _style(ax, "Training loss")
    ax.plot(g, [r["policy"] for r in rows], color=S1, lw=2, label="policy (cross-entropy)")
    ax.plot(g, [r["value"] for r in rows], color=S2, lw=2, label="value (MSE)")
    ax.set_xlabel("generation")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    _style(ax, "Self-play outcomes")
    ax.plot(g, [100 * r["first_wins"] for r in rows], color=S1, lw=2, label="first player wins")
    ax.plot(g, [100 * r["draws"] for r in rows], color=S3, lw=2, label="draws")
    ax.set_ylabel("% of self-play games"); ax.set_xlabel("generation")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.text(0.99, 0.01, f"{games[-1]:,} self-play games in total", ha="right", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training.png"), dpi=150)
    plt.close(fig)


def _pick(games, opening, side):
    """The longest game from this opening (the most informative curve)."""
    gs = [gm for gm in games if gm["opening"] == opening and gm["agent_side"] == side]
    return max(gs, key=lambda gm: len(gm["moves"])) if gs else None


def confidence(results: str, out: str):
    games = json.load(open(os.path.join(results, "solver_games.json")))
    picks = [("", 1, "Agent moves first (empty board)"),
             ("1", 2, "Opponent opens on the edge (col 1)"),
             ("3", 2, "Opponent opens on col 3"),
             ("4", 2, "Opponent opens in the centre")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), facecolor=SURF, sharey=True)
    for ax, (op, side, title) in zip(axes, picks):
        gm = _pick(games, op, side)
        _style(ax, title)
        if gm is None:
            continue
        ps = gm["positions"]
        x = [p["ply"] + 1 for p in ps]
        truth = [p["true"] for p in ps]
        ax.step(x, [50 * (t + 1) for t in truth], where="mid", color=MUTED, lw=1.5, ls="--",
                label="ground truth (solver)")
        ax.plot(x, [50 * (p["agent_value"] + 1) for p in ps], color=S1, lw=2, marker="o", ms=4,
                label="agent's win confidence")
        res = {1: "agent won", 0: "draw", -1: "agent lost"}[gm["result"]]
        theo = {1: "win", 0: "draw", -1: "loss"}[gm["theoretical"]]
        ax.text(0.02, 0.04, f"theory: {theo} · result: {res}", transform=ax.transAxes, color=INK, fontsize=9)
        ax.set_ylim(-4, 104); ax.set_xlabel("move number")
    axes[0].set_ylabel("agent win confidence (%)\n100 = win, 50 = draw, 0 = loss")
    axes[0].legend(frameon=False, fontsize=8.5, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confidence_games.png"), dpi=150)
    plt.close(fig)

    # calibration: average confidence by move number, split by true value
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURF)
    _style(ax, "Does the confidence match reality? Average over all games vs. the solver")
    for tv, col, lab in ((1, WIN, "positions that are won"), (0, DRAW, "drawn"), (-1, LOSS, "lost")):
        by = {}
        for gm in games:
            for p in gm["positions"]:
                by.setdefault(p["ply"] // 4 * 4, []).append((p["true"], p["agent_value"]))
        xs, ys = [], []
        for k in sorted(by):
            vals = [50 * (v + 1) for t, v in by[k] if t == tv]
            if len(vals) >= 3:
                xs.append(k + 2); ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_ylim(0, 100); ax.set_xlabel("move number"); ax.set_ylabel("mean win confidence (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    training(a.run, a.out)
    if os.path.exists(os.path.join(a.results, "solver_games.json")):
        confidence(a.results, a.out)
    print("plots written to", a.out)
