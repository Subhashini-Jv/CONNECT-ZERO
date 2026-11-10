"""Bitboard Connect Four engine (7 columns x 6 rows).

Layout: bit index = col * 7 + row, row 0 at the bottom. Bit 6 of each column
is an always-empty sentinel, which makes shift-based line detection safe.
`cur` holds the stones of the side to move, `mask` holds every stone.
"""
from __future__ import annotations

import numpy as np

WIDTH, HEIGHT = 7, 6
H1 = HEIGHT + 1
CELLS = WIDTH * HEIGHT

BOTTOM = [1 << (c * H1) for c in range(WIDTH)]
TOP = [1 << (HEIGHT - 1 + c * H1) for c in range(WIDTH)]
COLUMN = [((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH)]
BOTTOM_MASK = sum(BOTTOM)
BOARD_MASK = BOTTOM_MASK * ((1 << HEIGHT) - 1)


def has_four(p: int) -> bool:
    for d in (1, H1, HEIGHT, HEIGHT + 2):
        m = p & (p >> d)
        if m & (m >> (2 * d)):
            return True
    return False


def winning_spots(p: int, mask: int) -> int:
    """Empty cells that would complete four-in-a-row for stones `p`."""
    r = (p << 1) & (p << 2) & (p << 3)
    for d in (H1, HEIGHT, HEIGHT + 2):
        t = (p << d) & (p << (2 * d))
        r |= t & (p << (3 * d))
        r |= t & (p >> d)
        t = (p >> d) & (p >> (2 * d))
        r |= t & (p << d)
        r |= t & (p >> (3 * d))
    return r & (BOARD_MASK ^ mask)


class Board:
    __slots__ = ("cur", "mask", "moves", "history")

    def __init__(self, cur: int = 0, mask: int = 0, moves: int = 0, history=None):
        self.cur, self.mask, self.moves = cur, mask, moves
        self.history = list(history) if history else []

    # ----------------------------------------------------------- basics
    @classmethod
    def from_moves(cls, seq) -> "Board":
        """Build from a move list or a string of 1-based columns, e.g. '4453'."""
        b = cls()
        if isinstance(seq, str):
            seq = [int(ch) - 1 for ch in seq.strip()]
        for c in seq:
            if not b.can_play(c) or b.is_terminal():
                raise ValueError(f"illegal move {c + 1} in sequence")
            b.play(c)
        return b

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.cur, b.mask, b.moves, b.history = self.cur, self.mask, self.moves, self.history[:]
        return b

    def can_play(self, c: int) -> bool:
        return 0 <= c < WIDTH and not (self.mask & TOP[c])

    def legal_moves(self) -> list[int]:
        return [c for c in range(WIDTH) if not (self.mask & TOP[c])]

    def legal_mask(self) -> np.ndarray:
        return np.array([not (self.mask & TOP[c]) for c in range(WIDTH)], dtype=bool)

    def move_bit(self, c: int) -> int:
        return (self.mask + BOTTOM[c]) & COLUMN[c]

    def is_winning_move(self, c: int) -> bool:
        return has_four(self.cur | self.move_bit(c))

    def winning_moves(self) -> list[int]:
        spots = winning_spots(self.cur, self.mask) & ((self.mask + BOTTOM_MASK) & BOARD_MASK)
        return [c for c in range(WIDTH) if spots & COLUMN[c]]

    def play(self, c: int) -> None:
        self.cur ^= self.mask
        self.mask |= self.mask + BOTTOM[c]
        self.moves += 1
        self.history.append(c)

    def last_player_won(self) -> bool:
        """True if the player who just moved has four in a row."""
        return has_four(self.cur ^ self.mask)

    def is_full(self) -> bool:
        return self.moves >= CELLS

    def is_terminal(self) -> bool:
        return self.moves > 0 and (self.last_player_won() or self.is_full())

    def result(self) -> float | None:
        """Terminal value from the side to move: -1 lost, 0 draw, None ongoing."""
        if self.moves > 0 and self.last_player_won():
            return -1.0
        if self.is_full():
            return 0.0
        return None

    def key(self) -> int:
        return self.cur + self.mask

    # ----------------------------------------------------------- tensors
    def planes(self) -> np.ndarray:
        """(3, 6, 7) float32: own stones, opponent stones, constant ones."""
        out = np.empty((3, HEIGHT, WIDTH), dtype=np.float32)
        out[0] = _bits_to_grid(self.cur)
        out[1] = _bits_to_grid(self.cur ^ self.mask)
        out[2] = 1.0
        return out

    def grid(self) -> np.ndarray:
        """(6, 7) int grid, row 0 = TOP row for display. 1 = first player, 2 = second."""
        first = self.cur if self.moves % 2 == 0 else self.cur ^ self.mask
        second = first ^ self.mask
        g = _bits_to_grid(first).astype(np.int8) + 2 * _bits_to_grid(second).astype(np.int8)
        return g[::-1]

    def mirrored(self) -> "Board":
        b = Board()
        for c in self.history:
            b.play(WIDTH - 1 - c)
        return b

    def __str__(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[int(v)] for v in row) for row in self.grid()]
        return "\n".join(rows + ["1 2 3 4 5 6 7"])

    def move_string(self) -> str:
        return "".join(str(c + 1) for c in self.history)


_SHIFT = np.arange(64, dtype=np.uint64)


def _bits_to_grid(x: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(x.to_bytes(8, "little"), dtype=np.uint8), bitorder="little")
    return bits[:49].reshape(WIDTH, H1)[:, :HEIGHT].T.astype(np.float32)  # (row, col), row 0 bottom


def tactical_moves(b: Board) -> tuple[list[int], float | None]:
    """One-ply tactics shared by search and play.

    Returns (moves, proven_value). If the side to move can win at once,
    proven_value is +1 and moves lists the winning columns. If every move
    loses within one reply, proven_value is -1. Otherwise moves are the
    columns that do not hand the opponent an immediate win.
    """
    poss = (b.mask + BOTTOM_MASK) & BOARD_MASK
    mine = winning_spots(b.cur, b.mask) & poss
    if mine:
        return [c for c in range(WIDTH) if mine & COLUMN[c]], 1.0
    opp_win = winning_spots(b.cur ^ b.mask, b.mask)
    forced = poss & opp_win
    if forced:
        if forced & (forced - 1):
            return b.legal_moves(), -1.0
        poss = forced
    safe = poss & ~(opp_win >> 1)
    if not safe:
        return [c for c in range(WIDTH) if poss & COLUMN[c]], -1.0
    return [c for c in range(WIDTH) if safe & COLUMN[c]], None
