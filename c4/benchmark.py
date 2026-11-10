"""Benchmark a trained agent against the exact solver and simple baselines.

  python -m c4.benchmark runs/main/latest.pt --out results/

Produces results/benchmark.json (+ per-game records used by c4.plots).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

from . import net as netmod
from .agent import Agent
from .book import Book
from .evaluate import network_accuracy, search_accuracy
from .game import WIDTH, Board, tactical_moves
from .testset import load as load_testset

BOOK_PATH = "data/solver_book.json"
# Known theory (Allis 1988, Allen 1988): from the empty board the first player
# wins only with the centre column; columns 3 and 5 draw; the rest lose.
EMPTY_BOARD = [-1, -1, 0, 1, 0, -1, -1]


# ------------------------------------------------------------------ players
def solver_move(book: Book, b: Board, rng: random.Random) -> int:
    """A perfect player: always keeps the best game-theoretic result. Among
    equally good moves it prefers the fastest win / slowest loss (exact scores,
    when they can be computed within a node budget), then moves that do not
    lose at once, then a random one."""
    from . import solver
    sc = book.analyze(b, weak=True)
    best = max(v for v in sc if v is not None)
    cands = [c for c, v in enumerate(sc) if v == best]
    if len(cands) > 1:
        ex = solver.analyze_budget(b, 50_000_000, weak=False)
        if ex is not None:
            top = max(ex[c] for c in cands)
            cands = [c for c in cands if ex[c] == top]
        else:
            safe = [c for c in cands if c in tactical_moves(b)[0]]
            cands = safe or cands
    return rng.choice(cands)


def random_move(b: Board, rng) -> int:
    return rng.choice(b.legal_moves())


def tactical_move(b: Board, rng) -> int:
    """Wins when it can, blocks when it must, avoids moves that lose at once."""
    moves, _ = tactical_moves(b)
    return rng.choice(moves)


def rollout_mcts_move(b: Board, rng, sims: int = 2000) -> int:
    """Plain UCT with random rollouts (no network) as a classical baseline."""

    stats = {}

    def rollout(x: Board) -> float:
        x = x.copy()
        sign = 1.0
        while True:
            r = x.result()
            if r is not None:
                return sign * r
            moves, _ = tactical_moves(x)
            x.play(rng.choice(moves))
            sign = -sign

    def sim(x: Board) -> float:
        r = x.result()
        if r is not None:
            return r
        key = x.key()
        if key not in stats:
            stats[key] = [0, {}]
            return rollout(x)
        node = stats[key]
        moves = x.legal_moves()
        best, bm = -1e9, moves[0]
        for m in moves:
            n, w = node[1].get(m, (0, 0.0))
            u = 1e9 if n == 0 else w / n + 1.4 * math.sqrt(math.log(node[0] + 1) / n)
            if u > best:
                best, bm = u, m
        y = x.copy(); y.play(bm)
        v = -sim(y)
        n, w = node[1].get(bm, (0, 0.0))
        node[1][bm] = (n + 1, w + v)
        node[0] += 1
        return v

    wins = [c for c in b.winning_moves()]
    if wins:
        return wins[0]
    for _ in range(sims):
        sim(b)
    node = stats[b.key()]
    return max(node[1], key=lambda m: node[1][m][0])


# ------------------------------------------------------------------ games
def play_vs_solver(agent: Agent, book: Book, opening: str, agent_side: int, seed: int) -> dict:
    """Play from `opening` with the agent as player `agent_side` (1 = first)."""
    rng = random.Random(seed)
    b = Board.from_moves(opening)
    start_value = _value(book, b)  # for side to move
    rec = {"opening": opening, "agent_side": agent_side, "positions": []}
    while b.result() is None:
        side = 1 if b.moves % 2 == 0 else 2
        if side == agent_side:
            info = agent.analyse(b)
            mv = info["move"]
            sc = _scores(book, b)
            best = max(v for v in sc if v is not None)
            rec["positions"].append({"ply": b.moves, "who": "agent", "agent_value": info["value"],
                                     "net_value": info["net_value"], "true": best,
                                     "move": mv, "optimal": sc[mv] == best})
        else:
            sc = _scores(book, b)
            best = max(v for v in sc if v is not None)
            mv = solver_move(book, b, rng)
            info = agent.analyse(b, sims=agent.sims // 2)  # agent's view of the position, for the plot
            rec["positions"].append({"ply": b.moves, "who": "solver", "agent_value": -info["value"],
                                     "net_value": -info["net_value"], "true": -best, "move": mv})
        b.play(mv)
    r = b.result()
    winner = 0 if r == 0 else (1 if (b.moves - 1) % 2 == 0 else 2)
    first_to_move = 1 if Board.from_moves(opening).moves % 2 == 0 else 2
    theo_first = start_value if first_to_move == 1 else -start_value  # value for player 1
    theo_agent = theo_first if agent_side == 1 else -theo_first
    got = 0 if winner == 0 else (1 if winner == agent_side else -1)
    rec.update(moves=b.move_string(), winner=winner, theoretical=theo_agent, result=got,
               held=got >= theo_agent)
    return rec


def _scores(book: Book, b: Board):
    if b.moves == 0:
        return list(EMPTY_BOARD)
    return book.analyze(b, weak=True)


def _value(book: Book, b: Board) -> int:
    return max(v for v in _scores(book, b) if v is not None)


def play_vs_baseline(agent: Agent, opp, n_games: int, seed: int, book: Book | None = None) -> dict:
    rng = random.Random(seed)
    res = {"win": 0, "draw": 0, "loss": 0, "theoretically_lost_starts": 0, "losses_from_won_or_drawn_starts": 0}
    for g in range(n_games):
        agent_side = 1 + (g % 2)
        b = Board()
        # a couple of random opening plies so games differ
        for _ in range(2):
            b.play(rng.choice(b.legal_moves()))
        start = _value(book, b) if book else None  # for player 1 (to move)
        start_agent = None if start is None else (start if agent_side == 1 else -start)
        if start_agent == -1:
            res["theoretically_lost_starts"] += 1
        while b.result() is None:
            side = 1 if b.moves % 2 == 0 else 2
            b.play(agent(b) if side == agent_side else opp(b, rng))
        r = b.result()
        winner = 0 if r == 0 else (1 if (b.moves - 1) % 2 == 0 else 2)
        outcome = "draw" if winner == 0 else ("win" if winner == agent_side else "loss")
        res[outcome] += 1
        if outcome == "loss" and start_agent is not None and start_agent >= 0:
            res["losses_from_won_or_drawn_starts"] += 1
    return res


# ------------------------------------------------------------------ workers
_AGENT = None
_BOOK = None


def _init(ckpt, sims, guard):
    global _AGENT, _BOOK
    torch.set_num_threads(1)
    _AGENT = Agent(netmod.load(ckpt), sims=sims, guard=guard)
    _BOOK = Book(BOOK_PATH)


def _job(task):
    kind = task[0]
    t = time.time()
    if kind == "solver":
        _, opening, side, seed = task
        out = play_vs_solver(_AGENT, _BOOK, opening, side, seed)
    else:
        _, name, n, seed = task
        opp = {"random": random_move, "tactical": tactical_move,
               "uct-rollouts-2000": rollout_mcts_move}[name]
        out = {"baseline": name, **play_vs_baseline(_AGENT, opp, n, seed, _BOOK)}
    out["seconds"] = round(time.time() - t, 1)
    return out, {k: v for k, v in _BOOK.data.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--testset", default="data/testset.json")
    ap.add_argument("--extra-testset", default="data/testset_selfplay.json")
    ap.add_argument("--skip-games", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--only-baselines", action="store_true")
    ap.add_argument("--skip-testsets", action="store_true")
    ap.add_argument("--guard", action="store_true", help="benchmark Zero + Proof mode")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.workers)
    net = netmod.load(a.ckpt)
    report = {"checkpoint": a.ckpt, "sims": a.sims, "mode": "zero+proof" if a.guard else "zero"}

    # 1) solver-labelled positions
    for name, path in (("random_play", a.testset), ("strong_play", a.extra_testset)):
        if not os.path.exists(path) or a.skip_testsets:
            continue
        ts = load_testset(path)
        r = {"positions": len(ts), **network_accuracy(net, ts)}
        for s in (50, 200, a.sims):
            r[f"search_acc_{s}"] = search_accuracy(net, ts, sims=s)
        report[f"testset_{name}"] = r
        print(name, r, flush=True)

    if not a.skip_games:
        tasks = [] if a.only_baselines else [("solver", "", 1, s) for s in range(6)]                       # agent first, empty board
        if not a.only_baselines:
            tasks += [("solver", str(c), 2, s) for c in range(1, 8) for s in range(2)]  # agent second after each opening
            tasks += [("solver", f"{c1}{c2}", 1, 0) for c1 in range(1, 8) for c2 in range(1, 8)]  # all 2-ply openings
        if not a.skip_baselines:
            tasks += [("base", "random", 50, 1), ("base", "tactical", 50, 2), ("base", "uct-rollouts-2000", 20, 3)]
        ctx = mp.get_context("fork")
        games, baselines = [], []
        book = Book(BOOK_PATH)
        with ctx.Pool(a.workers, initializer=_init, initargs=(a.ckpt, a.sims, a.guard)) as pool:
            for out, data in pool.imap_unordered(_job, tasks):
                book.data.update(data)
                book.save()
                if "baseline" in out:
                    baselines.append(out)
                    print("baseline", out, flush=True)
                else:
                    games.append(out)
                    print("game", out["opening"] or "(empty)", "side", out["agent_side"], "theory", out["theoretical"],
                          "result", out["result"], "held" if out["held"] else "DROPPED", out["seconds"], "s", flush=True)
        if a.only_baselines:
            with open(os.path.join(a.out, "baselines.json"), "w") as f:
                json.dump(baselines, f, indent=2)
            print(json.dumps(baselines, indent=2))
            return
        with open(os.path.join(a.out, "solver_games.json"), "w") as f:
            json.dump(games, f)
        agent_moves = [p for g in games for p in g["positions"] if p["who"] == "agent"]
        report["vs_solver"] = {
            "games": len(games),
            "held_theoretical_result": sum(g["held"] for g in games),
            "agent_moves": len(agent_moves),
            "optimal_moves": sum(p["optimal"] for p in agent_moves),
            "by_group": {
                "agent_first_empty_board": _summ([g for g in games if g["opening"] == ""]),
                "agent_second_each_opening": _summ([g for g in games if len(g["opening"]) == 1]),
                "all_49_two_ply_openings": _summ([g for g in games if len(g["opening"]) == 2]),
            },
            "dropped": [{"opening": g["opening"], "side": g["agent_side"], "moves": g["moves"],
                         "theory": g["theoretical"], "result": g["result"]} for g in games if not g["held"]],
        }
        report["vs_baselines"] = baselines
    with open(os.path.join(a.out, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report.get("vs_solver", {}), indent=2))


def _summ(gs):
    return {"games": len(gs), "held": sum(g["held"] for g in gs),
            "wins": sum(g["result"] == 1 for g in gs), "draws": sum(g["result"] == 0 for g in gs),
            "losses": sum(g["result"] == -1 for g in gs),
            "theory_wins": sum(g["theoretical"] == 1 for g in gs),
            "theory_draws": sum(g["theoretical"] == 0 for g in gs),
            "theory_losses": sum(g["theoretical"] == -1 for g in gs)}


if __name__ == "__main__":
    main()
