"""Pre-compute solver results for all positions with 1-2 stones (slow; run once)."""
import sys
import time

from .book import Book, _canon
from .game import Board

book = Book("data/solver_book.json")
seen = []
for s in [str(c) for c in range(1, 8)] + [f"{a}{b}" for a in range(1, 8) for b in range(1, 8)]:
    k, _ = _canon(s)
    if k not in seen:
        seen.append(k)
for k in seen:
    t = time.time()
    sc = book.analyze(Board.from_moves(k))
    print(k, sc, f"{time.time() - t:.0f}s", flush=True)
