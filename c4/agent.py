"""High-level player built on the network + MCTS."""
from __future__ import annotations

import numpy as np

from . import mcts
from .game import WIDTH, Board, tactical_moves
from .mcts import Evaluator, Node, SearchConfig


class Agent:
    """Network + MCTS player.

    guard=True turns on "Zero + Proof" mode: whenever the position is in the
    proof book or the exact solver can settle it within `budget` search nodes,
    the agent only plays moves that are proven to keep the best result
    (preferring the move its own search liked). Otherwise it is pure MCTS.
    """

    def __init__(self, net, sims: int = 400, cfg: SearchConfig | None = None,
                 guard: bool = False, budget: int = 30_000_000, book_path: str = "data/proof_book.json"):
        self.cfg = cfg or SearchConfig()
        self.sims = sims
        self.evaluator = Evaluator(net)
        self.guard, self.budget = guard, budget
        self.book = None
        if guard:
            import json, os
            self.book = {}
            for path in (book_path, "data/solver_book.json"):
                if os.path.exists(path):
                    for k, v in json.load(open(path)).items():
                        if k.startswith("w:"):
                            self.book[k[2:]] = v

    def proof(self, board: Board):
        """(scores, from_book): exact win/draw/loss per column, or (None, False)."""
        from . import solver
        from .book import _canon
        if board.moves == 0:
            return [-1, -1, 0, 1, 0, -1, -1], True
        key, mirrored = _canon(board.move_string())
        if self.book is not None and key in self.book:
            sc = self.book[key]
            return (sc[::-1] if mirrored else list(sc)), True
        return solver.analyze_budget(board, self.budget), False

    def analyse(self, board: Board, sims: int | None = None) -> dict:
        info = self._search(board, sims)
        if self.guard and info.get("proven") is None:
            sc, from_book = self.proof(board)
            if sc is not None:
                best = max(v for v in sc if v is not None)
                cands = [c for c, v in enumerate(sc) if v == best]
                info["proof"] = best
                info["proof_scores"] = sc
                if from_book:  # fixed rule so the book covers every line it can reach
                    mv = max(cands, key=lambda c: info["prior"][c])
                    if mv != info["move"]:
                        info["overruled"] = info["move"]
                    info["move"] = mv
                elif info["move"] not in cands:
                    info["overruled"] = info["move"]
                    info["move"] = max(cands, key=lambda c: (info["visits"][c], np.nan_to_num(info["prior"][c])))
                info["value"] = float(best)
        return info

    def _search(self, board: Board, sims: int | None = None) -> dict:
        """Search `board`; returns move, visit distribution, per-move values and win chance."""
        sims = self.sims if sims is None else sims
        moves, proven = tactical_moves(board)
        logits, v = self.evaluator([board])
        prior = np.full(WIDTH, np.nan, dtype=np.float32)
        legal = board.legal_moves()
        lg = logits[0][legal]
        p = np.exp(lg - lg.max()); p /= p.sum()
        prior[legal] = p
        info = {"prior": prior, "net_value": float(v[0]), "proven": proven}
        if proven is not None or len(moves) == 1:
            # tactics decide: win now, block, or (lost) pick the network's favourite
            move = max(moves, key=lambda c: prior[c])
            visits = np.zeros(WIDTH, np.float32); visits[move] = 1
            value = proven if proven is not None else float(v[0])
            q = np.full(WIDTH, np.nan, np.float32); q[move] = value
            info.update(move=move, visits=visits, policy=visits, q=q, value=value)
            return info
        root = Node()
        mcts.search([root], [board], self.evaluator, self.cfg, sims=sims)
        visits = root.visits()
        q = root.child_q()
        move = int(np.argmax(visits + 1e-3 * np.nan_to_num(q, nan=-2)))
        value = float(q[move])
        info.update(move=move, visits=visits, policy=visits / visits.sum(), q=q, value=value)
        return info

    def __call__(self, board: Board) -> int:
        return self.analyse(board)["move"]


def win_probability(value: float) -> float:
    """Map a value in [-1, 1] to a 0-100% "win confidence" (draw = 50%)."""
    return 50.0 * (value + 1.0)
