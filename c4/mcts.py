"""Batched PUCT Monte Carlo tree search.

Many independent trees (one per game) are searched in lock-step so that every
leaf reached in a round is evaluated by the network in a single batch. That
is what makes self-play usable on a CPU.

Values are always stored from the point of view of the player to move at the
node that owns the edge statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .game import WIDTH, Board, tactical_moves


@dataclass
class SearchConfig:
    sims: int = 200
    c_puct: float = 2.0
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 1.0
    noise_frac: float = 0.25
    tactics: bool = True  # prune moves that lose at once, stop on immediate wins


class Node:
    __slots__ = ("moves", "P", "N", "W", "children", "n", "terminal", "expanded")

    def __init__(self):
        self.moves: list[int] = []
        self.P = None
        self.N = None
        self.W = None
        self.children = None
        self.n = 0
        self.terminal = None  # proven value for the side to move, or None
        self.expanded = False

    def q(self) -> float:
        return sum(self.W) / self.n if self.n else 0.0

    def visits(self) -> np.ndarray:
        v = np.zeros(WIDTH, dtype=np.float32)
        if self.expanded:
            for i, a in enumerate(self.moves):
                v[a] = self.N[i]
        return v

    def child_q(self) -> np.ndarray:
        """Mean value of each move for the side to move here (nan if unvisited)."""
        q = np.full(WIDTH, np.nan, dtype=np.float32)
        if self.expanded:
            for i, a in enumerate(self.moves):
                if self.N[i]:
                    q[a] = self.W[i] / self.N[i]
        return q


class Evaluator:
    """Wraps a network: batch of boards -> (priors over 7 columns, values)."""

    def __init__(self, net, device="cpu"):
        self.net, self.device = net, device
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, boards: list[Board]):
        self.calls += len(boards)
        x = torch.from_numpy(np.stack([b.planes() for b in boards])).to(self.device)
        logits, v = self.net(x)
        return logits.float().cpu().numpy(), v.float().cpu().numpy()


def _prepare(node: Node, board: Board, cfg: SearchConfig) -> bool:
    """Set terminal status / candidate moves. Returns True if NN eval is needed."""
    r = board.result()
    if r is not None:
        node.terminal = r
        return False
    if cfg.tactics:
        moves, proven = tactical_moves(board)
        if proven is not None:
            node.terminal = proven
            node.moves = moves
            return False
        node.moves = moves
    else:
        node.moves = board.legal_moves()
    return True


def _expand(node: Node, logits: np.ndarray):
    lg = logits[node.moves]
    p = np.exp(lg - lg.max())
    p /= p.sum()
    k = len(node.moves)
    node.P = p.tolist()
    node.N = [0] * k
    node.W = [0.0] * k
    node.children = [None] * k
    node.expanded = True


def _select(node: Node, cfg: SearchConfig) -> int:
    sq = cfg.c_puct * math.sqrt(max(node.n, 1))
    fpu = (sum(node.W) / node.n if node.n else 0.0) - cfg.fpu_reduction
    best, best_i = -1e9, 0
    N, W, P = node.N, node.W, node.P
    for i in range(len(N)):
        n = N[i]
        u = (W[i] / n if n else fpu) + P[i] * sq / (1 + n)
        if u > best:
            best, best_i = u, i
    return best_i


def add_noise(node: Node, cfg: SearchConfig, rng: np.random.Generator):
    if node.expanded and len(node.moves) > 1 and cfg.noise_frac > 0:
        eta = rng.dirichlet([cfg.dirichlet_alpha] * len(node.moves))
        node.P = [(1 - cfg.noise_frac) * p + cfg.noise_frac * e for p, e in zip(node.P, eta)]


def search(roots: list[Node], boards: list[Board], evaluator, cfg: SearchConfig,
           sims: list[int] | int | None = None, rng=None, noise: bool = False):
    """Run simulations on several trees at once (in place)."""
    n_games = len(roots)
    sims = [cfg.sims] * n_games if sims is None else ([sims] * n_games if isinstance(sims, int) else sims)

    # make sure roots are expanded first
    need = [i for i in range(n_games) if not roots[i].expanded and roots[i].terminal is None]
    need = [i for i in need if _prepare(roots[i], boards[i], cfg)]
    if need:
        logits, _ = evaluator([boards[i] for i in need])
        for j, i in enumerate(need):
            _expand(roots[i], logits[j])
    if noise:
        for r in roots:
            add_noise(r, cfg, rng)

    target = max(sims) if sims else 0
    for s in range(target):
        leaves, leaf_boards, paths = [], [], []
        for g in range(n_games):
            root = roots[g]
            if s >= sims[g] or root.terminal is not None:
                continue
            node, b, path = root, boards[g].copy(), []
            while True:
                if node.terminal is not None:
                    _backup(path, node.terminal)
                    break
                if not node.expanded:
                    leaves.append(node); leaf_boards.append(b); paths.append(path)
                    break
                i = _select(node, cfg)
                path.append((node, i))
                b.play(node.moves[i])
                child = node.children[i]
                if child is None:
                    child = node.children[i] = Node()
                    if not _prepare(child, b, cfg):
                        _backup(path, child.terminal)
                        break
                node = child
        if leaves:
            logits, values = evaluator(leaf_boards)
            for j, leaf in enumerate(leaves):
                _expand(leaf, logits[j])
                _backup(paths[j], float(values[j]))


def _backup(path, v: float):
    # v is from the perspective of the side to move at the leaf
    for node, i in reversed(path):
        v = -v
        node.N[i] += 1
        node.W[i] += v
        node.n += 1


def child(node: Node, move: int) -> Node:
    """Subtree after `move` (for tree reuse); a fresh node if unexplored."""
    if node.expanded and move in node.moves:
        c = node.children[node.moves.index(move)]
        if c is not None:
            return c
    return Node()
