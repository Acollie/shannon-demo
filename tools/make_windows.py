"""Held-out windows for export_model.py / build_data.py: the validation split of a training shard (public FineWeb-Edu).

  python tools/make_windows.py --shannon-src ~/shannon --shard ~/shannon/data/shards/fw600m --out heldout.npz

Writes src/tgt [N,128] token ids (uint16) and tb [N,128] target byte counts, using the training scaffold's own WindowSet
split, so these windows were never trained on.
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shannon-src", required=True)
ap.add_argument("--shard", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lin", type=int, default=128)
ap.add_argument("--lout", type=int, default=128)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.shannon_src))
from data import WindowSet  # noqa: E402

ws = WindowSet(a.shard, a.lin, a.lout, "val")
src, tgt, tb = ws.get(np.arange(len(ws)))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, src=src.numpy().astype(np.uint16), tgt=tgt.numpy().astype(np.uint16), tb=tb.numpy().astype(np.uint16))
print(a.out, len(ws), "windows")
