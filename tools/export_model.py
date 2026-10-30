"""Export a Shannon checkpoint to three browser-friendly ONNX graphs.

  encoder.onnx   input_ids [1, L] (L multiple of R, <= lin)        -> codes [1, L/R]          (frozen MiniLM + segmenter + VQ argmin)
  ideas.onnx     codes [1, M]                                      -> cross_k, cross_v [Ld, H, M, dh]
                 (codebook lookup -> idea-level transformer -> per-decoder-layer cross-attention keys/values)
  decoder.onnx   token [1], pos [1], k_cache, v_cache [Ld, H, T, dh], cross_k, cross_v
                 -> logits [V], k_cache_out, v_cache_out                (one decoding step with a fixed-size KV cache)

The decoder sees ONLY what ideas.onnx makes from the codes, exactly like ShannonModel.decode_from_codes.

The model code is imported (not copied) from the training scaffold given by --shannon-src (needs models.py).

  python tools/export_model.py --shannon-src /path/to/shannon --ckpt runs/X/cooldown_1.0.pt \
      --windows heldout.npz --out models/<model-id> [--quant auto|int8|fp32]

--windows: npz with src/tgt [N,128] token ids and tb [N,128] target byte counts (held-out windows) for the BPB check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN2 = math.log(2.0)


def load(src_dir, ckpt, config_path=None):
    sys.path.insert(0, os.path.abspath(src_dir))
    from models import ModelConfig, build_model  # noqa: E402
    cp = config_path or os.path.join(os.path.dirname(os.path.abspath(ckpt)), "config.json")
    rc = json.load(open(cp))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" in sd and "cfg" in sd:
        rc.setdefault("cfg", sd["cfg"])
        sd = sd["model"]
    cfg = ModelConfig.from_dict(rc["cfg"])
    assert cfg.arm == "shannon", "only the Shannon arm can be exported (the decoder must read ideas only)"
    m = build_model(cfg)
    m.load_state_dict(sd)
    return m.eval(), rc


# ------------------------------------------------------------------------------------------------ export wrappers
class EncoderToCodes(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.R = m.cfg.R

    def forward(self, ids):
        h = self.m.encoder(ids)                                  # (1, L, 384)
        seg = h.reshape(1, -1, self.R * h.shape[-1])
        z = F.normalize(self.m.adapter(seg), dim=-1)[0]          # (M, d_vq)
        cb = self.m.vq.codebook
        dist = cb.pow(2).sum(-1)[None] - 2 * z @ cb.t()           # |z|^2 is constant per row
        return dist.argmin(-1)[None]                              # (1, M)


class IdeasToCross(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        c = m.cfg
        self.H, self.dh = c.n_heads, c.d_model // c.n_heads

    def forward(self, codes):
        z_q = F.embedding(codes, self.m.vq.codebook)
        mem = self.m.ideas_from_zq(z_q)[0]                        # (M, d)
        M = mem.shape[0]
        ks, vs = [], []
        for b in self.m.decoder.stack.blocks:
            ks.append(b.ca.k(mem).view(M, self.H, self.dh).transpose(0, 1))
            vs.append(b.ca.v(mem).view(M, self.H, self.dh).transpose(0, 1))
        return torch.stack(ks), torch.stack(vs)


class DecoderStep(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        c = m.cfg
        self.H, self.dh, self.T = c.n_heads, c.d_model // c.n_heads, c.lout

    def forward(self, tok, pos, kc, vc, ck, cv):
        d = self.m.decoder
        H, dh, T = self.H, self.dh, self.T
        x = d.emb(tok) + d.pos(pos)                               # (1, D)
        ar = torch.arange(T)
        oh = (ar == pos).float()[None, :, None]                   # (1, T, 1)
        bias = torch.where(ar <= pos, 0.0, -1e9)[None, None, :]    # (1, 1, T)
        scale = 1.0 / math.sqrt(dh)
        nk, nv = [], []
        for l, b in enumerate(d.stack.blocks):
            h = b.n1(x)
            q = b.sa.q(h).view(H, 1, dh)
            k = b.sa.k(h).view(H, 1, dh)
            v = b.sa.v(h).view(H, 1, dh)
            kl = kc[l] * (1 - oh) + k * oh                        # (H, T, dh)
            vl = vc[l] * (1 - oh) + v * oh
            nk.append(kl)
            nv.append(vl)
            a = torch.softmax(q @ kl.transpose(1, 2) * scale + bias, -1) @ vl   # (H, 1, dh)
            x = x + b.sa.o(a.reshape(1, H * dh))
            h = b.nx(x)
            q = b.ca.q(h).view(H, 1, dh)
            a = torch.softmax(q @ ck[l].transpose(1, 2) * scale, -1) @ cv[l]
            x = x + b.ca.o(a.reshape(1, H * dh))
            x = x + b.down(F.gelu(b.up(b.n2(x))))
        x = d.stack.norm(x)
        logits = (x @ d.emb.weight.t())[0]
        return logits, torch.stack(nk), torch.stack(nv)


# ------------------------------------------------------------------------------------------------ reference / checks
@torch.no_grad()
def torch_reference(m, src, tgt):
    from models import shift_right  # noqa: E402
    codes = m.encode_to_codes(src)
    logits = m.decode_from_codes(codes, shift_right(tgt))
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
    return codes, nll


def onnx_window(sess, cfg, src_ids, tgt_ids, bos=101):
    enc, ide, dec = sess
    L, H, dh, T = cfg.n_dec_layers, cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.lout
    codes = enc.run(None, {"input_ids": src_ids[None].astype(np.int64)})[0]
    ck, cv = ide.run(None, {"codes": codes})
    kc = np.zeros((L, H, T, dh), np.float32)
    vc = np.zeros_like(kc)
    prev = bos
    nll = np.zeros(len(tgt_ids))
    for t, y in enumerate(tgt_ids):
        lg, kc, vc = dec.run(None, {"token": np.array([prev], np.int64), "pos": np.array([t], np.int64),
                                    "k_cache": kc, "v_cache": vc, "cross_k": ck, "cross_v": cv})
        lg = lg.astype(np.float64)
        mx = lg.max()
        nll[t] = mx + math.log(np.exp(lg - mx).sum()) - lg[y]
        prev = int(y)
    return codes[0], nll


def fp16_weights(src, dst, min_numel=1024):
    """Store large float initializers as fp16 + a Cast back to fp32 (compute stays fp32; ORT folds the Casts at load)."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    g = onnx.load(src)
    new_inits, casts = [], []
    for init in g.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_numel:
            arr = numpy_helper.to_array(init).astype(np.float16)
            h = numpy_helper.from_array(arr, init.name + "__fp16")
            new_inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            new_inits.append(init)
    del g.graph.initializer[:]
    g.graph.initializer.extend(new_inits)
    nodes = list(g.graph.node)
    del g.graph.node[:]
    g.graph.node.extend(casts + nodes)
    onnx.save(g, dst)


def sessions(paths):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return [ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shannon-src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="auto", choices=["auto", "int8", "fp32"], help="decoder-side graphs (ideas, decoder)")
    ap.add_argument("--enc-precision", default="fp16w", choices=["fp16w", "int8", "fp32"],
                    help="encoder: fp16w = fp16-stored weights, fp32 compute (keeps idea codes identical to PyTorch far more often than int8)")
    ap.add_argument("--n-check", type=int, default=12)
    ap.add_argument("--tol", type=float, default=0.01, help="max relative BPB change allowed for int8")
    a = ap.parse_args()

    torch.manual_seed(0)
    m, rc = load(a.shannon_src, a.ckpt, a.config)
    cfg = m.cfg
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_fp32")
    os.makedirs(tmp, exist_ok=True)
    L, H, dh, T, M = cfg.n_dec_layers, cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.lout, cfg.n_ideas

    ex = dict(opset_version=17, do_constant_folding=True, dynamo=False)
    with torch.no_grad():
        torch.onnx.export(EncoderToCodes(m), (torch.randint(1000, 30000, (1, cfg.lin)),), f"{tmp}/encoder.onnx",
                          input_names=["input_ids"], output_names=["codes"], dynamic_axes={"input_ids": {1: "L"}, "codes": {1: "M"}}, **ex)
        torch.onnx.export(IdeasToCross(m), (torch.randint(0, cfg.K, (1, M)),), f"{tmp}/ideas.onnx",
                          input_names=["codes"], output_names=["cross_k", "cross_v"],
                          dynamic_axes={"codes": {1: "M"}, "cross_k": {2: "M"}, "cross_v": {2: "M"}}, **ex)
        z = torch.zeros(L, H, T, dh)
        torch.onnx.export(DecoderStep(m), (torch.tensor([101]), torch.tensor([0]), z, z, torch.zeros(L, H, M, dh), torch.zeros(L, H, M, dh)),
                          f"{tmp}/decoder.onnx", input_names=["token", "pos", "k_cache", "v_cache", "cross_k", "cross_v"],
                          output_names=["logits", "k_cache_out", "v_cache_out"],
                          dynamic_axes={"cross_k": {2: "M"}, "cross_v": {2: "M"}}, **ex)
    names = ["encoder", "ideas", "decoder"]

    # step-wise wrapper == teacher-forced full decoder (pure torch)
    W = np.load(a.windows)
    src = torch.from_numpy(W["src"][: a.n_check].astype(np.int64))
    tgt = torch.from_numpy(W["tgt"][: a.n_check].astype(np.int64))
    tb = W["tb"][: a.n_check].astype(np.float64)
    codes_t, nll_t = torch_reference(m, src, tgt)
    bpb_t = nll_t.sum(1).numpy() / LN2 / tb.sum(1)
    with torch.no_grad():
        ck, cv = IdeasToCross(m)(codes_t[:1])
        kc = vc = torch.zeros(L, H, T, dh)
        prev, nll_s = 101, []
        for t in range(T):
            lg, kc, vc = DecoderStep(m)(torch.tensor([prev]), torch.tensor([t]), kc, vc, ck, cv)
            nll_s.append(float(torch.logsumexp(lg, 0) - lg[tgt[0, t]]))
            prev = int(tgt[0, t])
        assert torch.equal(EncoderToCodes(m)(src[:1]), codes_t[:1])
    step_err = float(np.abs(np.array(nll_s) - nll_t[0].numpy()).max())
    print(f"torch step-wrapper vs full decoder: max |dNLL| = {step_err:.2e}")

    def check(paths, tag):
        S = sessions(paths)
        agree, bpb, t0 = [], [], time.time()
        for i in range(a.n_check):
            c, nll = onnx_window(S, cfg, src[i].numpy(), tgt[i].numpy())
            agree.append(float((c == codes_t[i].numpy()).mean()))
            bpb.append(nll.sum() / LN2 / tb[i].sum())
        bpb = np.array(bpb)
        r = {"code_agreement": float(np.mean(agree)), "bpb_torch": float(bpb_t.mean()), "bpb_onnx": float(bpb.mean()),
             "bpb_rel_diff": float(bpb.mean() / bpb_t.mean() - 1), "max_window_abs_diff": float(np.abs(bpb - bpb_t).max()),
             "seconds": time.time() - t0}
        print(tag, json.dumps(r))
        return r

    report = {"n_windows": a.n_check, "torch_stepwise_max_abs_dnll": step_err}
    report["fp32"] = check([f"{tmp}/{n}.onnx" for n in names], "fp32")
    final = {n: f"{tmp}/{n}.onnx" for n in names}
    precision = "fp32"
    if a.quant in ("auto", "int8"):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        qdir = os.path.join(a.out, "_int8")
        os.makedirs(qdir, exist_ok=True)
        for n in names:
            quantize_dynamic(f"{tmp}/{n}.onnx", f"{qdir}/{n}.onnx", weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gemm", "Gather"], per_channel=False)
        report["int8_all"] = check([f"{qdir}/{n}.onnx" for n in names], "int8 (all three graphs)")
        enc_path = {"int8": f"{qdir}/encoder.onnx", "fp32": f"{tmp}/encoder.onnx"}.get(a.enc_precision)
        if a.enc_precision == "fp16w":
            fp16_weights(f"{tmp}/encoder.onnx", f"{qdir}/encoder_fp16w.onnx")
            enc_path = f"{qdir}/encoder_fp16w.onnx"
        mixed = [enc_path, f"{qdir}/ideas.onnx", f"{qdir}/decoder.onnx"]
        report["shipped_candidate"] = check(mixed, f"encoder {a.enc_precision} + int8 ideas/decoder")
        if a.quant == "int8" or abs(report["shipped_candidate"]["bpb_rel_diff"]) <= a.tol:
            final = dict(zip(names, mixed))
            precision = f"encoder {a.enc_precision}, ideas+decoder int8"
    import shutil
    sizes = {}
    for n, p in final.items():
        shutil.copy(p, os.path.join(a.out, f"{n}.onnx"))
        sizes[n] = os.path.getsize(p)
        assert sizes[n] < 50e6, f"{n}.onnx is {sizes[n]/1e6:.1f} MB (> 50 MB)"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(os.path.join(a.out, "_int8"), ignore_errors=True)
    report["precision"] = precision
    report["sizes_bytes"] = sizes
    arch = {"lin": cfg.lin, "lout": cfg.lout, "R": cfg.R, "K": cfg.K, "d_vq": cfg.d_vq, "vocab": cfg.vocab, "d_model": cfg.d_model,
            "n_heads": cfg.n_heads, "head_dim": dh, "n_dec_layers": cfg.n_dec_layers, "n_idea_layers": cfg.n_idea_layers,
            "n_ideas": cfg.n_ideas, "bos_id": 101, "min_gen_id": 999}
    json.dump({"arch": arch, "params": rc.get("params"), "export_check": report}, open(os.path.join(a.out, "export.json"), "w"), indent=1)
    print(json.dumps({"precision": precision, "sizes_MB": {k: round(v / 1e6, 1) for k, v in sizes.items()}}))


if __name__ == "__main__":
    main()
