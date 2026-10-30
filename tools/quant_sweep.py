"""Try int8 quantisation variants of an fp32 export (left in <out>/_fp32 by export_model.py) and report BPB drift vs PyTorch.

  python tools/quant_sweep.py --shannon-src ~/shannon --ckpt runs/X/cooldown_1.0.pt --windows heldout.npz --out models/<id> \
      [--pick per_channel_nogather]   # copy that variant (+ fp16w encoder) to <out>/{encoder,ideas,decoder}.onnx and update export.json

Variants: per_channel, per_channel_nogather (embeddings stay fp32 -> fp16w), fp16w (no int8), mixed (per_channel ideas + fp16w decoder).
(export_model.py's own int8 is per-tensor.) For 20M every int8 decoder variant tried moved BPB 2.5-3.4%, so 20M ships "mixed" (0.02%).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_model import LN2, fp16_weights, load, onnx_window, sessions, torch_reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shannon-src", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--windows", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-check", type=int, default=24)
ap.add_argument("--variants", default="per_channel,per_channel_nogather,fp16w")
ap.add_argument("--pick", default="")
a = ap.parse_args()

m, rc = load(a.shannon_src, a.ckpt)
cfg = m.cfg
W = np.load(a.windows)
src = torch.from_numpy(W["src"][: a.n_check].astype(np.int64))
tgt = torch.from_numpy(W["tgt"][: a.n_check].astype(np.int64))
tb = W["tb"][: a.n_check].astype(np.float64)
codes_t, nll_t = torch_reference(m, src, tgt)
bpb_t = nll_t.sum(1).numpy() / LN2 / tb.sum(1)
fp = os.path.join(a.out, "_fp32")
qd = os.path.join(a.out, "_q")
os.makedirs(qd, exist_ok=True)
enc = os.path.join(qd, "encoder_fp16w.onnx")
if not os.path.exists(enc):
    fp16_weights(f"{fp}/encoder.onnx", enc)


def build(v):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    if v == "mixed":      # int8 per-channel ideas + fp16-stored decoder (the 20M decoder drifts ~3% BPB under any int8 variant tried)
        return [enc, build("per_channel")[1], build("fp16w")[2]]
    outs = []
    for n in ("ideas", "decoder"):
        dst = f"{qd}/{n}_{v}.onnx"
        if not os.path.exists(dst):
            if v == "fp16w":
                fp16_weights(f"{fp}/{n}.onnx", dst)
            else:
                ops = ["MatMul", "Gemm"] + ([] if v.endswith("nogather") else ["Gather"])
                tmp = dst + ".tmp.onnx"
                quantize_dynamic(f"{fp}/{n}.onnx", tmp, weight_type=QuantType.QInt8, op_types_to_quantize=ops,
                                 per_channel=v.startswith("per_channel"))
                if v.endswith("nogather"):
                    fp16_weights(tmp, dst)          # remaining fp32 tensors (embeddings) stored as fp16
                    os.remove(tmp)
                else:
                    os.replace(tmp, dst)
        outs.append(dst)
    return [enc] + outs


def check(paths):
    S = sessions(paths)
    agree, bpb = [], []
    for i in range(a.n_check):
        c, nll = onnx_window(S, cfg, src[i].numpy(), tgt[i].numpy())
        agree.append(float((c == codes_t[i].numpy()).mean()))
        bpb.append(nll.sum() / LN2 / tb[i].sum())
    bpb = np.array(bpb)
    return {"code_agreement": float(np.mean(agree)), "bpb_torch": float(bpb_t.mean()), "bpb_onnx": float(bpb.mean()),
            "bpb_rel_diff": float(bpb.mean() / bpb_t.mean() - 1), "max_window_abs_diff": float(np.abs(bpb - bpb_t).max()),
            "sizes_bytes": {n: os.path.getsize(p) for n, p in zip(("encoder", "ideas", "decoder"), paths)}}


res = {}
for v in [x for x in a.variants.split(",") if x]:
    res[v] = check(build(v))
    print(v, json.dumps(res[v]), flush=True)
if a.pick:
    paths = build(a.pick)
    r = res.get(a.pick) or check(paths)
    for n, p in zip(("encoder", "ideas", "decoder"), paths):
        shutil.copy(p, os.path.join(a.out, f"{n}.onnx"))
    ep = os.path.join(a.out, "export.json")
    dh = cfg.d_model // cfg.n_heads
    arch = {"lin": cfg.lin, "lout": cfg.lout, "R": cfg.R, "K": cfg.K, "d_vq": cfg.d_vq, "vocab": cfg.vocab, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "head_dim": dh, "n_dec_layers": cfg.n_dec_layers, "n_idea_layers": cfg.n_idea_layers,
            "n_ideas": cfg.n_ideas, "bos_id": 101, "min_gen_id": 999}
    rep = {"n_windows": a.n_check, "variants": res, "shipped_candidate": r,
           "precision": {"fp16w": "fp16 weights (fp32 compute)", "mixed": "encoder+decoder fp16w, ideas int8 per-channel"}.get(a.pick, f"encoder fp16w, ideas+decoder int8 {a.pick.replace('_', ' ')}"),
           "sizes_bytes": r["sizes_bytes"]}
    json.dump({"arch": arch, "params": rc.get("params"), "export_check": rep}, open(ep, "w"), indent=1)
    print("picked", a.pick, json.dumps(r["sizes_bytes"]))
