"""Export a checkpoint for the browser: BatchNorm folded into convolutions,
float32 weights packed as base64 into a small JS file (window.C4_MODEL)."""
from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import torch

from . import net as netmod


def _fold(conv, bn):
    w = conv.weight.detach().double()
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    w = w * scale[:, None, None, None]
    b = bn.bias.detach().double() - bn.running_mean.double() * scale
    return w.float().numpy(), b.float().numpy()


def export(ckpt: str, out: str, meta: dict | None = None):
    net = netmod.load(ckpt)
    layers = []

    def add(name, w, b):
        layers.append((name, w, b))

    add("stem", *_fold(net.stem[0], net.stem[1]))
    for i, blk in enumerate(net.tower):
        add(f"b{i}a", *_fold(blk.c1, blk.b1))
        add(f"b{i}b", *_fold(blk.c2, blk.b2))
    add("p_conv", *_fold(net.p_conv[0], net.p_conv[1]))
    add("p_fc", net.p_fc.weight.detach().numpy(), net.p_fc.bias.detach().numpy())
    add("v_conv", *_fold(net.v_conv[0], net.v_conv[1]))
    add("v_fc1", net.v_fc1.weight.detach().numpy(), net.v_fc1.bias.detach().numpy())
    add("v_fc2", net.v_fc2.weight.detach().numpy(), net.v_fc2.bias.detach().numpy())

    index, chunks, off = [], [], 0
    for name, w, b in layers:
        w = np.ascontiguousarray(w, dtype=np.float32); b = np.ascontiguousarray(b, dtype=np.float32)
        index.append({"name": name, "wshape": list(w.shape), "w": off, "b": off + w.size})
        chunks += [w.ravel(), b.ravel()]
        off += w.size + b.size
    blob = np.concatenate(chunks).astype("<f4").tobytes()
    model = {"config": net.config(), "layers": index, "meta": meta or {},
             "data": base64.b64encode(blob).decode()}
    with open(out, "w") as f:
        f.write("window.C4_MODEL = " + json.dumps(model) + ";\n")

    # self-check: reference outputs so the JS port can be verified
    from .game import Board
    b = Board.from_moves("4453")
    with torch.inference_mode():
        p, v = net(torch.from_numpy(b.planes()[None]))
    return {"moves": "4453", "logits": p[0].tolist(), "value": float(v[0])}


def export_proof(out: str = "web/proof.js", wasm: str = "web/build/solver.wasm",
                 books=("data/proof_book.json", "data/solver_book.json")):
    """Write the WebAssembly solver and the opening proof book for the browser."""
    import os
    import subprocess
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver.c")
    os.makedirs(os.path.dirname(wasm), exist_ok=True)
    subprocess.check_call(["clang", "--target=wasm32", "-O3", "-nostdlib", "-Wl,--no-entry",
                           "-Wl,--export=c4_analyze_budget", "-Wl,--export=c4_nodes", "-Wl,--export=c4_reset",
                           "-Wl,--export-memory", "-Wl,--initial-memory=33554432", "-o", wasm, src])
    book = {}
    for path in books:
        if os.path.exists(path):
            for k, v in json.load(open(path)).items():
                if k.startswith("w:"):
                    book[k[2:]] = v
    with open(out, "w") as f:
        f.write("window.C4_SOLVER = " + json.dumps(base64.b64encode(open(wasm, "rb").read()).decode()) + ";\n")
        f.write("window.C4_BOOK = " + json.dumps(book, separators=(",", ":")) + ";\n")
    return len(book)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="web/model.js")
    ap.add_argument("--proof", action="store_true", help="also build web/proof.js (needs clang with wasm32)")
    ap.add_argument("--meta", default="{}", help="JSON shown in the UI (e.g. benchmark numbers)")
    a = ap.parse_args()
    print(json.dumps(export(a.ckpt, a.out, json.loads(a.meta))))
    if a.proof:
        print("proof book positions:", export_proof())
