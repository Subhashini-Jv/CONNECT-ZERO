"""Disk-backed cache of solver results for early positions, where solving is slow.

Keys are move strings canonicalised under left-right mirroring, so a position
and its mirror image share one entry.
"""
from __future__ import annotations

import json
import os
import threading

from . import solver
from .game import WIDTH, Board

CACHE_PLIES = 14  # positions with fewer stones than this are cached


def _canon(moves: str) -> tuple[str, bool]:
    mir = "".join(str(WIDTH + 1 - int(ch)) for ch in moves)
    return (mir, True) if mir < moves else (moves, False)


class Book:
    def __init__(self, path: str = "data/solver_book.json"):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def analyze(self, b: Board, weak: bool = True) -> list[int | None]:
        moves = b.move_string()
        if b.moves >= CACHE_PLIES:
            return solver.analyze(b, weak)
        key, mirrored = _canon(moves)
        key = ("w:" if weak else "s:") + key
        if key not in self.data:
            canon = Board.from_moves(key[2:])
            self.data[key] = solver.analyze(canon, weak)
            self.save()
        sc = self.data[key]
        return sc[::-1] if mirrored else list(sc)

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, self.path)

    def value(self, b: Board) -> int:
        """Win/draw/loss (+1/0/-1) for the side to move."""
        r = b.result()
        if r is not None:
            return int(r)
        return max(v for v in self.analyze(b) if v is not None)
