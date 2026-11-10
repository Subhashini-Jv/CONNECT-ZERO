"""Accuracy of the network / search on solver-labelled positions."""
from __future__ import annotations

import numpy as np
import torch

from . import mcts
from .game import Board, tactical_moves
from .mcts import Evaluator, Node, SearchConfig


def _boards(testset):
    return [Board.from_moves(p["moves"]) for p in testset]


@torch.inference_mode()
def network_accuracy(net, testset) -> dict:
    """Raw policy (no search) move accuracy and value-head outcome accuracy."""
    boards = _boards(testset)
    x = torch.from_numpy(np.stack([b.planes() for b in boards]))
    logits, v = net(x)
    logits, v = logits.numpy(), v.numpy()
    ok_p = ok_v = 0
    for i, (b, p) in enumerate(zip(boards, testset)):
        legal = b.legal_moves()
        mv = max(legal, key=lambda c: logits[i][c])
        ok_p += mv in p["best"]
        pred = 1 if v[i] > 1 / 3 else (-1 if v[i] < -1 / 3 else 0)
        ok_v += pred == p["value"]
    n = len(testset)
    return {"policy_acc": ok_p / n, "value_acc": ok_v / n}


def search_accuracy(net, testset, sims: int = 200, cfg: SearchConfig | None = None, chunk: int = 250) -> float:
    """Move accuracy of the full agent (tactics + MCTS) with `sims` simulations."""
    cfg = cfg or SearchConfig()
    ev = Evaluator(net)
    boards = _boards(testset)
    picks = [None] * len(boards)
    todo = []
    for i, b in enumerate(boards):
        moves, proven = tactical_moves(b)
        if proven == 1.0 or len(moves) == 1:
            picks[i] = moves[0]
        else:
            todo.append(i)
    for s in range(0, len(todo), chunk):
        idx = todo[s:s + chunk]
        roots = [Node() for _ in idx]
        mcts.search(roots, [boards[i] for i in idx], ev, cfg, sims=sims)
        for i, r in zip(idx, roots):
            if r.terminal is not None:  # proven loss: anything goes
                picks[i] = r.moves[0] if r.moves else boards[i].legal_moves()[0]
            else:
                picks[i] = int(np.argmax(r.visits()))
    ok = sum(picks[i] in p["best"] for i, p in enumerate(testset))
    return ok / len(testset)
