"""Self-play game generation (runs inside worker processes)."""
from __future__ import annotations

import numpy as np
import torch

from . import mcts
from .game import WIDTH, Board, tactical_moves
from .mcts import Evaluator, Node, SearchConfig
from .net import C4Net


def play_games(state: dict, net_cfg: dict, n_games: int, cfg: SearchConfig, seed: int,
               parallel: int = 32, temp_plies: int = 10, fast_sims: int = 0, full_frac: float = 1.0):
    """Play `n_games` self-play games, `parallel` at a time.

    Returns (samples, records). Each sample is (planes, policy, z, q, use_policy).
    With fast_sims > 0, playout-cap randomisation is used: only a `full_frac`
    share of moves get the full search and contribute policy targets.
    """
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    net = C4Net(**net_cfg)
    net.load_state_dict(state)
    net.eval()
    ev = Evaluator(net)

    samples, records = [], []
    started = 0
    games = []  # each: dict(board, root, hist)

    def new_game():
        nonlocal started
        started += 1
        return {"board": Board(), "root": Node(), "hist": []}

    while started < min(parallel, n_games):
        games.append(new_game())

    while games:
        # 1) handle positions decided by tactics without search
        search_idx, full = [], []
        for gi, g in enumerate(games):
            moves, proven = tactical_moves(g["board"])
            if proven is not None or len(moves) == 1:
                g["forced"] = (moves, proven)
            else:
                g["forced"] = None
                search_idx.append(gi)
                full.append(fast_sims <= 0 or rng.random() < full_frac)
        if search_idx:
            roots = [games[i]["root"] for i in search_idx]
            boards = [games[i]["board"] for i in search_idx]
            sims = [cfg.sims if f else fast_sims for f in full]
            mcts.search(roots, boards, ev, cfg, sims=sims, rng=rng, noise=True)
        full_of = dict(zip(search_idx, full))

        # 2) choose moves
        finished = []
        for gi, g in enumerate(games):
            b = g["board"]
            if g["forced"] is not None:
                moves, proven = g["forced"]
                move = moves[rng.integers(len(moves))] if proven == -1.0 else moves[0]
                pi = np.zeros(WIDTH, np.float32); pi[move] = 1.0
                q = proven if proven is not None else None
                g["hist"].append((b.planes(), pi, b.moves, q, proven != -1.0))
            else:
                root = g["root"]
                visits = root.visits()
                pi = visits / visits.sum()
                if b.moves < temp_plies:
                    move = int(rng.choice(WIDTH, p=pi))
                else:
                    move = int(np.argmax(visits))
                g["hist"].append((b.planes(), pi, b.moves, root.q(), full_of[gi]))
            g["root"] = mcts.child(g["root"], move) if g["forced"] is None else Node()
            b.play(move)
            r = b.result()
            if r is not None:
                finished.append(gi)
                # r is from the side to move now; the last mover gets -r
                n = b.moves
                for planes, pi, ply, q, use_pi in g["hist"]:
                    z = r if (n - ply) % 2 == 0 else -r
                    samples.append((planes, pi, float(z), float(z if q is None else q), bool(use_pi)))
                winner = 0 if r == 0 else (1 if (n - 1) % 2 == 0 else 2)
                records.append({"moves": b.move_string(), "winner": winner})
        for gi in reversed(finished):
            games.pop(gi)
            if started < n_games:
                games.append(new_game())
    return samples, records
