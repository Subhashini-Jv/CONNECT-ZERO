"""AlphaZero-style training loop: self-play -> replay buffer -> SGD -> repeat."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from . import net as netmod
from .evaluate import network_accuracy, search_accuracy
from .mcts import SearchConfig
from .selfplay import play_games
from .testset import load as load_testset


def to_tensors(samples):
    planes = torch.from_numpy(np.stack([s[0] for s in samples]))
    pi = torch.from_numpy(np.stack([s[1] for s in samples]))
    z = torch.tensor([s[2] for s in samples], dtype=torch.float32)
    q = torch.tensor([s[3] for s in samples], dtype=torch.float32)
    use = torch.tensor([s[4] for s in samples], dtype=torch.float32)
    return planes, pi, z, q, use


def train_steps(net, opt, buffer, steps, batch, q_weight, rng):
    planes, pi, z, q, use = buffer
    n = planes.shape[0]
    net.train()
    tot = {"loss": 0.0, "policy": 0.0, "value": 0.0}
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, size=batch))
        x, p, zz, qq, u = planes[idx], pi[idx], z[idx], q[idx], use[idx]
        flip = torch.from_numpy(rng.random(batch) < 0.5)  # mirror augmentation
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        p = torch.where(flip[:, None], p.flip(-1), p)
        target_v = (1 - q_weight) * zz + q_weight * qq
        logits, v = net(x)
        lp = -(p * F.log_softmax(logits, dim=1)).sum(1)
        lp = (lp * u).sum() / u.sum().clamp(min=1)
        lv = F.mse_loss(v, target_v)
        loss = lp + lv
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot["loss"] += loss.item(); tot["policy"] += lp.item(); tot["value"] += lv.item()
    net.eval()
    return {k: v / max(steps, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--games", type=int, default=400, help="self-play games per generation")
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--fast-sims", type=int, default=0, help="playout-cap randomisation (0 = off)")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=32, help="concurrent games per worker")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reuse", type=float, default=4.0, help="how often each new sample is trained on")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--window-max", type=int, default=20, help="replay window in generations")
    ap.add_argument("--q-weight", type=float, default=0.5, help="value target = mix of z and root Q")
    ap.add_argument("--testset", default="data/testset.json")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.workers)
    rng = np.random.default_rng(0)
    testset = load_testset(a.testset) if os.path.exists(a.testset) else None

    net = netmod.C4Net(a.channels, a.blocks)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    start = 0
    ck_path = os.path.join(a.out, "latest.pt")
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = netmod.C4Net(**ck["config"]); net.load_state_dict(ck["state"])
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        opt.load_state_dict(ck["opt"])
        start = ck["gen"] + 1
    net.eval()
    cfg = SearchConfig(sims=a.sims)
    windows: deque = deque()
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    games_path = os.path.join(a.out, "games.jsonl")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    ctx = mp.get_context("fork")
    total_games = 0
    with ctx.Pool(a.workers) as pool:
        for gen in range(start, start + a.gens):
            t0 = time.time()
            state = {k: v.clone() for k, v in net.state_dict().items()}
            per = [a.games // a.workers + (i < a.games % a.workers) for i in range(a.workers)]
            jobs = [(state, net.config(), per[i], cfg, gen * 100 + i, a.parallel, 10,
                     a.fast_sims, 0.25 if a.fast_sims else 1.0) for i in range(a.workers)]
            results = pool.starmap(play_games, jobs)
            samples = [s for r in results for s in r[0]]
            records = [g for r in results for g in r[1]]
            total_games += len(records)
            t_play = time.time() - t0

            windows.append(samples)
            win_len = min(a.window_max, 4 + gen // 2)
            while len(windows) > win_len:
                windows.popleft()
            buffer = to_tensors([s for w in windows for s in w])

            # learning-rate schedule: decay over the planned run
            frac = gen / max(start + a.gens - 1, 1)
            for g in opt.param_groups:
                g["lr"] = a.lr * (0.1 ** frac)
            steps = max(1, int(len(samples) * a.reuse / a.batch))
            losses = train_steps(net, opt, buffer, steps, a.batch, a.q_weight, rng)
            t_train = time.time() - t0 - t_play

            m = {"gen": gen, "games": len(records), "samples": len(samples), "buffer": buffer[0].shape[0],
                 "first_wins": sum(r["winner"] == 1 for r in records) / len(records),
                 "draws": sum(r["winner"] == 0 for r in records) / len(records),
                 "avg_len": float(np.mean([len(r["moves"]) for r in records])),
                 "lr": opt.param_groups[0]["lr"], "steps": steps, **losses,
                 "t_play": round(t_play, 1), "t_train": round(t_train, 1)}
            if testset:
                m.update(network_accuracy(net, testset))
                if a.eval_every and (gen % a.eval_every == a.eval_every - 1 or gen == start + a.gens - 1):
                    m["search_acc"] = search_accuracy(net, testset, sims=a.eval_sims)
            m["t_total"] = round(time.time() - t0, 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            with open(games_path, "a") as f:
                for r in records:
                    f.write(json.dumps({"gen": gen, **r}) + "\n")
            netmod.save(net, ck_path, {"opt": opt.state_dict(), "gen": gen})
            netmod.save(net, os.path.join(a.out, f"gen{gen:03d}.pt"), {"gen": gen})
            print(json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
