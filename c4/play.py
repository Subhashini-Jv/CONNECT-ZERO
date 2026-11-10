"""Play Connect Four against the trained agent in the terminal.

  python -m c4.play                      # you move first
  python -m c4.play --ai-first --sims 1600
  python -m c4.play --guard              # Zero + Proof (unbeatable) mode
  python -m c4.play --coach              # the exact solver grades each of your moves
  python -m c4.play --watch              # agent vs perfect solver
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

from . import net as netmod
from .agent import Agent
from .game import WIDTH, Board

GOLD, ROSE, DIM, BOLD, RESET = "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def draw(b: Board, color: bool = True):
    sym = {0: f"{DIM}.{RESET}" if color else ".", 1: f"{GOLD}●{RESET}" if color else "X",
           2: f"{ROSE}●{RESET}" if color else "O"}
    for row in b.grid():
        print("  " + " ".join(sym[int(v)] for v in row))
    print("  " + " ".join(str(c + 1) for c in range(WIDTH)))


def bar(pct: float, width: int = 30) -> str:
    k = int(round(pct / 100 * width))
    return "[" + "#" * k + "-" * (width - k) + f"] {pct:5.1f}%"


def show_analysis(info: dict):
    vis = info["visits"]; tot = vis.sum() or 1
    cells = []
    for c in range(WIDTH):
        q = info["q"][c]
        cells.append(" -- " if np.isnan(q) else f"{50 * (q + 1):3.0f}%")
    print("  move value: " + " ".join(cells))
    print("  search    : " + " ".join(f"{100 * v / tot:3.0f}%" for v in vis))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--ai-first", action="store_true")
    ap.add_argument("--guard", action="store_true", help="Zero + Proof mode (proof book + exact solver)")
    ap.add_argument("--coach", action="store_true", help="grade your moves with the exact solver")
    ap.add_argument("--watch", action="store_true", help="watch the agent play the perfect solver")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    color = not a.no_color and sys.stdout.isatty()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    agent = Agent(netmod.load(a.model), sims=a.sims, guard=a.guard)
    b = Board()
    human = 2 if a.ai_first else 1
    curve = []

    if a.watch:
        from .benchmark import solver_move
        from .book import Book
        book, rng = Book(), random.Random()
        human = 2  # "human" seat is taken by the solver

    while b.result() is None:
        draw(b, color)
        side = 1 if b.moves % 2 == 0 else 2
        if side != human:
            info = agent.analyse(b)
            ai_win = 50 * (info["value"] + 1)
            curve.append(ai_win)
            show_analysis(info)
            proof = info.get("proof")
            tag = "" if proof is None else {1: "  [proven win]", 0: "  [proven draw]", -1: "  [proven loss]"}[int(proof)]
            print(f"  AI confidence {bar(ai_win)}  -> plays {info['move'] + 1}{tag}\n")
            b.play(info["move"])
            continue
        if a.watch:
            mv = solver_move(book, b, rng)
            print(f"  solver plays {mv + 1}\n")
            b.play(mv)
            continue
        while True:
            s = input(f"  your move (1-7, h = hint, q = quit): ").strip().lower()
            if s == "q":
                return
            if s == "h":
                info = agent.analyse(b)
                print(f"  hint: column {info['move'] + 1}")
                continue
            if s.isdigit() and b.can_play(int(s) - 1):
                mv = int(s) - 1
                break
            print("  not a legal column")
        if a.coach:
            from . import solver
            sc = solver.analyze(b, weak=True) if b.moves >= 6 else None
            if sc is not None:
                best = max(v for v in sc if v is not None)
                verdict = {1: "winning", 0: "drawing", -1: "losing"}
                if sc[mv] == best:
                    print(f"  coach: perfect move (position stays {verdict[best]})")
                else:
                    good = [c + 1 for c, v in enumerate(sc) if v == best]
                    print(f"  coach: mistake, {verdict[best]} -> {verdict[sc[mv]]}. Best: {good}")
        b.play(mv)
    draw(b, color)
    r = b.result()
    if r == 0:
        print("\n  Draw.")
    else:
        last = 1 if (b.moves - 1) % 2 == 0 else 2
        print("\n  " + ("You win!" if last == human and not a.watch else ("Solver wins." if a.watch and last == human else "AI wins.")))
    if curve:
        print("  AI confidence over the game: " + " ".join(f"{v:.0f}" for v in curve))
    print("  moves:", b.move_string())


if __name__ == "__main__":
    main()
