"""Residual policy/value network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import HEIGHT, WIDTH


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class C4Net(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.channels, self.blocks = channels, blocks
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.p_fc = nn.Linear(4 * HEIGHT * WIDTH, WIDTH)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.v_fc1 = nn.Linear(2 * HEIGHT * WIDTH, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = self.tower(self.stem(x))
        p = self.p_fc(self.p_conv(h).flatten(1))
        v = F.relu(self.v_fc1(self.v_conv(h).flatten(1)))
        return p, torch.tanh(self.v_fc2(v)).squeeze(1)

    def config(self):
        return {"channels": self.channels, "blocks": self.blocks}


def save(net: C4Net, path: str, extra: dict | None = None):
    torch.save({"config": net.config(), "state": net.state_dict(), **(extra or {})}, path)


def load(path: str) -> C4Net:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = C4Net(**ck["config"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net
