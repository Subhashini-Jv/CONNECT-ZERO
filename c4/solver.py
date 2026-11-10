"""Python bindings to the exact C solver (compiled on first use)."""
from __future__ import annotations

import ctypes
import os
import subprocess

from .game import Board, WIDTH

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "solver.c")
_LIB = os.path.join(_HERE, "libc4solver.so")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
        cc = os.environ.get("CC", "cc")
        tmp = _LIB + f".{os.getpid()}.tmp"
        subprocess.check_call([cc, "-O3", "-shared", "-fPIC", "-o", tmp, _SRC])
        os.replace(tmp, _LIB)  # atomic, so running processes keep their copy
    lib = ctypes.CDLL(_LIB)
    u64 = ctypes.c_uint64
    lib.c4_solve.argtypes = [u64, u64, ctypes.c_int, ctypes.c_int]
    lib.c4_solve.restype = ctypes.c_int
    lib.c4_analyze.argtypes = [u64, u64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.c4_nodes.restype = ctypes.c_ulonglong
    lib.c4_analyze_budget.argtypes = [u64, u64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_ulonglong]
    lib.c4_analyze_budget.restype = ctypes.c_int
    _lib = lib
    return lib


def solve(b: Board, weak: bool = False) -> int:
    """Exact score for the side to move (see solver.c), or -1/0/1 if weak."""
    r = b.result()
    if r is not None:  # game already over: side to move has lost, or it is a draw
        return 0 if r == 0 else (-1 if weak else -((43 - b.moves) // 2))
    return _load().c4_solve(b.cur, b.mask, b.moves, int(weak))


def analyze(b: Board, weak: bool = False) -> list[int | None]:
    """Score of each column for the side to move (None = illegal)."""
    out = (ctypes.c_int * WIDTH)()
    _load().c4_analyze(b.cur, b.mask, b.moves, int(weak), out)
    return [None if v == -1000 else v for v in out]


def analyze_budget(b: Board, budget: int, weak: bool = True) -> list[int | None] | None:
    """Like analyze() but returns None if not solved within `budget` search nodes."""
    out = (ctypes.c_int * WIDTH)()
    ok = _load().c4_analyze_budget(b.cur, b.mask, b.moves, int(weak), out, budget)
    return [None if v == -1000 else v for v in out] if ok else None


def best_moves(b: Board, weak: bool = True) -> list[int]:
    """Columns that achieve the best (theoretical) result."""
    sc = analyze(b, weak)
    best = max(v for v in sc if v is not None)
    return [c for c, v in enumerate(sc) if v == best]


def outcome(score: int) -> int:
    return (score > 0) - (score < 0)


def nodes() -> int:
    return _load().c4_nodes()
