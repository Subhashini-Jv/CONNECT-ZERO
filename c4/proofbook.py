"""Build the proof book used by "Zero + Proof" mode.

Walks every line the opponent can choose for the first plies, with the agent
answering by a fixed rule (the network's favourite among proven-optimal
moves), and stores exact win/draw/loss scores for every position where the
agent must decide. Beyond the book the agent uses the budgeted solver, and
when that is too slow, its own search.

  python -m c4.proofbook models/best.pt --first-plies 6 --second-plies 5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import net as netmod
from . import solver
from .book import _canon
from .game import Board, tactical_moves

EMPTY = [-1, -1, 0, 1, 0, -1, -1]


def _solve(key: str):
    return key, solver.analyze(Board.from_moves(key), weak=True)


def book_move(scores, prior) -> int:
    best = max(v for v in scores if v is not None)
    cands = [c for c, v in enumerate(scores) if v == best]
    return max(cands, key=lambda c: prior[c])


@torch.inference_mode()
def priors(net, boards):
    x = torch.from_numpy(np.stack([b.planes() for b in boards]))
    logits, _ = net(x)
    return logits.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="data/proof_book.json")
    ap.add_argument("--first-plies", type=int, default=6, help="last ply (as first player) covered")
    ap.add_argument("--second-plies", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    net = netmod.load(a.ckpt)
    book: dict[str, list] = {}
    try:
        book = {k[2:]: v for k, v in json.load(open("data/solver_book.json")).items() if k.startswith("w:")}
    except FileNotFoundError:
        pass
    out: dict[str, list] = {}

    with mp.get_context("fork").Pool(a.workers) as pool:
        for side, last in ((1, a.first_plies), (2, a.second_plies)):
            frontier = [""] if side == 1 else [str(c) for c in range(1, 8)]
            while frontier:
                t = time.time()
                boards = [Board.from_moves(s) for s in frontier]
                # agent decisions: skip ones decided by tactics (win now / forced block)
                need = sorted({_canon(s)[0] for s, b in zip(frontier, boards)
                               if b.moves > 0 and _canon(s)[0] not in book})
                for k, sc in pool.imap_unordered(_solve, need):
                    book[k] = sc
                pri = priors(net, boards)
                nxt = []
                for s, b, p in zip(frontier, boards, pri):
                    if b.moves == 0:
                        sc = EMPTY
                    else:
                        k, mir = _canon(s)
                        sc = book[k][::-1] if mir else book[k]
                        out[k] = book[k]
                    moves, proven = tactical_moves(b)
                    if proven == 1.0:
                        continue
                    mv = book_move(sc, p)
                    b.play(mv)
                    if b.result() is not None or b.moves + 1 > last:
                        continue
                    for c in b.legal_moves():
                        bb = b.copy(); bb.play(c)
                        if bb.result() is None:
                            nxt.append(bb.move_string())
                print(f"side {side} ply {boards[0].moves}: {len(frontier)} positions, "
                      f"{len(need)} solved in {time.time() - t:.0f}s", flush=True)
                frontier = sorted(set(nxt))
                with open(a.out, "w") as f:
                    json.dump({"w:" + k: v for k, v in out.items()}, f)
    print(f"{len(out)} positions in {a.out}")


if __name__ == "__main__":
    main()
