"""Solver-labelled benchmark positions.

Positions come from uniformly random games (like the classic Connect Four
solver test suites). Only "decision" positions are kept: at least one legal
move would change the game-theoretic result, so a move can be wrong.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import random

from . import solver
from .game import Board


def _random_position(rng: random.Random, lo: int, hi: int) -> Board | None:
    n = rng.randint(lo, hi)
    b = Board()
    for _ in range(n):
        if b.is_terminal():
            return None
        b.play(rng.choice(b.legal_moves()))
    return None if b.is_terminal() else b


def _label(args):
    seed, count, lo, hi = args
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        b = _random_position(rng, lo, hi)
        if b is None:
            continue
        scores = solver.analyze(b, weak=True)
        legal = [v for v in scores if v is not None]
        if len(set(legal)) < 2:
            continue
        best = max(legal)
        out.append({"moves": b.move_string(), "scores": scores, "value": best,
                    "best": [c for c, v in enumerate(scores) if v == best]})
    return out


def _label_games(args):
    seqs, = args
    out = []
    for seq in seqs:
        b = Board.from_moves(seq)
        scores = solver.analyze(b, weak=True)
        legal = [v for v in scores if v is not None]
        if len(set(legal)) < 2:
            continue
        best = max(legal)
        out.append({"moves": seq, "scores": scores, "value": best,
                    "best": [c for c, v in enumerate(scores) if v == best]})
    return out


def build_from_games(games_path: str, path: str, n: int = 600, min_gen: int = 0, lo: int = 8, hi: int = 34,
                     seed: int = 11, workers: int = 4):
    """Positions taken from strong self-play games (closer to real play than random games)."""
    rng = random.Random(seed)
    games = [json.loads(l) for l in open(games_path)]
    games = [g for g in games if g["gen"] >= min_gen]
    seen, cands = set(), []
    rng.shuffle(games)
    for g in games:
        m = g["moves"]
        if len(m) - 1 < lo:
            continue
        k = rng.randint(lo, min(hi, len(m) - 1))
        if m[:k] in seen:
            continue
        seen.add(m[:k]); cands.append(m[:k])
        if len(cands) >= int(n * 1.6):
            break
    chunks = [cands[i::40] for i in range(40)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_label_games, [(c,) for c in chunks])
    data = [p for part in parts for p in part][:n]
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def build(path: str, n: int = 1000, lo: int = 8, hi: int = 34, seed: int = 7, workers: int = 4):
    chunks = 40
    per = n // chunks
    with mp.Pool(workers) as pool:
        parts = pool.map(_label, [(seed * 1000 + i, per, lo, hi) for i in range(chunks)])
    data = [p for part in parts for p in part]
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def load(path: str):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build a solver-labelled test set")
    ap.add_argument("--out", default="data/testset.json")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = build(a.out, a.n, workers=a.workers)
    print(f"wrote {len(d)} positions to {a.out}")
