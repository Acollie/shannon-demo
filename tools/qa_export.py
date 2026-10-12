"""Export the three Shannon QA models to ONNX and check int8 quantisation against PyTorch.

  embed   sentence-transformers/all-MiniLM-L6-v2  -> mean-pooled, L2-normalised sentence vector
  reader  deepset/minilm-uncased-squad2           -> start_logits, end_logits
  yesno   treaty-graph data/hybrid/yesno_x         -> logits (contradiction, entailment, neutral)

Run on the GPU PC, CPU only:
  nice -n 19 ~/demo-venv/bin/python qa_export.py --src ~/qa-demo/src --out ~/qa-demo/out --docs ~/qa-demo/docs

Writes <out>/<name>.fp32.onnx, <out>/<name>.int8.onnx and <out>/quant_check.json.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer


class Embed(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        msk = attention_mask.unsqueeze(-1).float()
        v = (h * msk).sum(1) / msk.sum(1)
        return torch.nn.functional.normalize(v, dim=-1)


class Reader(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        o = self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return o.start_logits, o.end_logits


class YesNo(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def export(mod, tok, path, outs, sample):
    """torch.export-based exporter: the legacy tracer bakes the sample's mask shapes into the graph
    (transformers 5 masking code) and drifts badly on other lengths."""
    import onnx
    x = tok(*sample, return_tensors="pt", padding=True, truncation=True, max_length=64)
    names = ["input_ids", "attention_mask", "token_type_ids"]
    args = tuple(x[k] for k in names)
    B, S = torch.export.Dim("batch"), torch.export.Dim("seq", max=512)
    prog = torch.onnx.export(mod, args, input_names=names, dynamic_shapes=tuple({0: B, 1: S} for _ in names),
                             dynamo=True, opset_version=18, external_data=False)
    prog.optimize()
    m = prog.model_proto
    print(path.name, "outputs", [o.name for o in m.graph.output], "(in order:", outs, ")", flush=True)
    onnx.save(m, str(path))


def fp16_weights(src, dst, min_numel=1024):
    """Store large float initializers as fp16 + a Cast back to fp32 (compute stays fp32)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = onnx.load(str(src))
    new_inits, casts = [], []
    for init in g.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_numel:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            new_inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            new_inits.append(init)
    del g.graph.initializer[:]
    g.graph.initializer.extend(new_inits)
    nodes = list(g.graph.node)
    del g.graph.node[:]
    g.graph.node.extend(casts + nodes)
    onnx.save(g, str(dst))


# variant -> (op types to int8, per-channel); everything left in fp32 is then stored as fp16
VARIANTS = {
    "int8": (["MatMul", "Gemm", "Gather"], False),           # plain dynamic int8 (all MatMuls), no fp16 step
    "fp16w": ([], False),
    "mm8": (["MatMul", "Gemm"], True),                         # weight MatMuls int8 per-channel, embeddings fp16
    "mm8emb8": (["MatMul", "Gemm", "Gather"], True),
    "emb8": (["Gather"], False),                               # only the embedding table int8, rest fp16
}


def quantize(src, dst, variant):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    ops, pc = VARIANTS[variant]
    m = onnx.load(str(src))           # dynamo's value_info can trip ORT's shape inference (DeBERTa); drop it
    if len(m.graph.value_info):
        m.graph.ClearField("value_info")
        src = Path(str(dst) + ".src.onnx")
        onnx.save(m, str(src))
    if variant == "int8":
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=False)
        if str(src).endswith(".src.onnx"):
            Path(src).unlink()
        return
    tmp = Path(str(dst) + ".tmp.onnx")
    if ops:
        quantize_dynamic(str(src), str(tmp), weight_type=QuantType.QInt8, per_channel=pc, op_types_to_quantize=ops,
                         extra_options={"MatMulConstBOnly": True})
    else:
        tmp.write_bytes(Path(src).read_bytes())
    fp16_weights(tmp, dst)
    tmp.unlink()
    if str(src).endswith(".src.onnx"):
        Path(src).unlink()


def top_spans(tok, q, par, st, en, n=6):
    f = tok(q, par, truncation="only_second", max_length=384, return_offsets_mapping=True)
    offs, tt = f["offset_mapping"], f["token_type_ids"]
    null = float(st[0] + en[0])
    ctx = [i for i, t in enumerate(tt) if t == 1 and offs[i][1] > 0]
    ts = sorted(ctx, key=lambda i: -st[i])[:12]
    te = sorted(ctx, key=lambda i: -en[i])[:12]
    c = sorted(((float(st[i] + en[j]), offs[i][0], offs[j][1]) for i in ts for j in te if i <= j < i + 30), reverse=True)
    out, seen = [], set()
    for sc, a_, b_ in c:
        s = par[a_:b_]
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append((s, sc - null))
        if len(out) >= n:
            break
    return out


def passages(text, max_words=120, min_words=12):
    out, carry = [], ""
    for m in re.finditer(r"(?s)(.+?)(?:\n\s*\n|\Z)", text):
        block = " ".join(m.group(1).split())
        if not block:
            continue
        if len(block.split()) < min_words:
            carry = (carry + " " + block).strip()
            continue
        cur, carry = carry, ""
        for s in re.split(r"(?<=[.!?;:])\s+", block):
            if cur and len((cur + " " + s).split()) > max_words:
                out.append(cur)
                cur = ""
            cur = (cur + " " + s).strip()
        if cur:
            out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True, help="dir with the example .txt files and examples.json")
    ap.add_argument("--only", default="embed,reader,yesno")
    ap.add_argument("--variants", default="int8,fp16w,mm8,mm8emb8,emb8")
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    ex = json.loads((a.docs / "examples.json").read_text())
    pars, qs = [], []
    for d in ex["docs"]:
        pars += passages((a.docs / Path(d["file"]).name).read_text())
        qs += [q["q"] for q in d["questions"]]
    print(f"{len(pars)} passages, {len(qs)} questions", flush=True)
    report = {}
    rng = np.random.default_rng(0)

    sessions = {}

    def ort_run(path, feeds):
        import onnxruntime as ort
        if path not in sessions:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            sessions[path] = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        s = sessions[path]
        names = {i.name for i in s.get_inputs()}
        return s.run(None, {k: v.numpy().astype(np.int64) for k, v in feeds.items() if k in names})

    for name in a.only.split(","):
        sessions.clear()
        if name == "embed":
            tok = AutoTokenizer.from_pretrained(a.src / "all-minilm")
            mod = Embed(AutoModel.from_pretrained(a.src / "all-minilm").eval())
            outs = ["embedding"]
            sample = (["a short test sentence", "another one"],)
        elif name == "reader":
            tok = AutoTokenizer.from_pretrained(a.src / "minilm-squad2")
            mod = Reader(AutoModelForQuestionAnswering.from_pretrained(a.src / "minilm-squad2").eval())
            outs = ["start_logits", "end_logits"]
            sample = (["who?", "what?"], ["a short test passage", "another passage here"])
        else:
            tok = AutoTokenizer.from_pretrained(a.src / "yesno_x")
            mod = YesNo(AutoModelForSequenceClassification.from_pretrained(a.src / "yesno_x").eval())
            outs = ["logits"]
            sample = (["a short test passage", "another passage here"], ["is it?", "does it?"])
        fp32 = a.out / f"{name}.fp32.onnx"
        if not (a.skip_export and fp32.exists()):
            with torch.no_grad():
                export(mod, tok, fp32, outs, sample)
        variants = {"fp32": fp32}
        for v in a.variants.split(","):
            p = a.out / f"{name}.{v}.onnx"
            if not p.exists():
                quantize(fp32, p, v)
            variants[v] = p
        # checks on real inputs, one at a time (as the browser runs them)
        rep = {v: {"mb": round(p.stat().st_size / 1e6, 1)} for v, p in variants.items()}
        if name == "embed":
            texts = pars + qs
            feeds = [tok([t], truncation=True, max_length=256, return_tensors="pt") for t in texts]
            with torch.no_grad():
                ref = np.stack([mod(f["input_ids"], f["attention_mask"], f["token_type_ids"])[0].numpy() for f in feeds])
            P, Q = ref[: len(pars)], ref[len(pars):]
            rank_ref = [list((-(P @ q)).argsort()[:3]) for q in Q]
            for v, p in variants.items():
                got = np.stack([ort_run(p, dict(f))[0][0] for f in feeds])
                gp, gq = got[: len(pars)], got[len(pars):]
                rk = [list((-(gp @ q)).argsort()[:3]) for q in gq]
                rep[v].update({"min_cos_vs_torch": float((got * ref).sum(1).min()),
                               "top3_same": float(np.mean([r == s for r, s in zip(rk, rank_ref)])),
                               "max_abs_cos_err": float(np.abs(gp @ gq.T - P @ Q.T).max())})
        elif name == "reader":
            pairs = [(q, pars[int(i)]) for q in qs for i in rng.choice(len(pars), 3, replace=False)]
            errs = {v: [] for v in variants}
            same = {v: [] for v in variants}
            spans_same = {v: [] for v in variants}
            marg = {v: [] for v in variants}
            for q, par in pairs:
                f = tok(q, par, truncation="only_second", max_length=384, return_tensors="pt")
                with torch.no_grad():
                    s0, e0 = (t[0].numpy() for t in mod(f["input_ids"], f["attention_mask"], f["token_type_ids"]))
                for v, p in variants.items():
                    s1, e1 = (t[0] for t in ort_run(p, dict(f)))
                    errs[v].append(float(max(np.abs(s1 - s0).max(), np.abs(e1 - e0).max())))
                    same[v].append(int(s1.argmax() == s0.argmax() and e1.argmax() == e0.argmax()))
                    r0, r1 = top_spans(tok, q, par, s0, e0), top_spans(tok, q, par, s1, e1)
                    spans_same[v].append(int([x for x, _ in r0] == [x for x, _ in r1]))
                    marg[v].append(abs(r0[0][1] - r1[0][1]) if r0 and r1 else 0.0)
            for v in variants:
                rep[v].update({"max_logit_err": max(errs[v]), "mean_logit_err": float(np.mean(errs[v])),
                               "argmax_same": float(np.mean(same[v])), "top6_spans_same": float(np.mean(spans_same[v])),
                               "max_best_margin_err": float(max(marg[v])), "mean_best_margin_err": float(np.mean(marg[v]))})
        else:
            pairs = [(pars[int(i)], q) for q in qs for i in rng.choice(len(pars), 3, replace=False)]
            dif = {v: [] for v in variants}
            flip = {v: [] for v in variants}
            for par, q in pairs:
                f = tok([par], [q], truncation=True, max_length=320, return_tensors="pt")
                with torch.no_grad():
                    p0 = mod(f["input_ids"], f["attention_mask"], f["token_type_ids"]).softmax(-1)[0].numpy()
                y0 = p0[1] / (p0[1] + p0[0] + 1e-9)
                for v, p in variants.items():
                    lg = ort_run(p, dict(f))[0][0]
                    pr = np.exp(lg - lg.max())
                    pr /= pr.sum()
                    y1 = pr[1] / (pr[1] + pr[0] + 1e-9)
                    dif[v].append(abs(float(y1 - y0)))
                    flip[v].append(int((y1 > 0.5) != (y0 > 0.5)))
            for v in variants:
                rep[v].update({"max_pyes_err": max(dif[v]), "mean_pyes_err": float(np.mean(dif[v])),
                               "flips": int(sum(flip[v])), "n": len(pairs)})
        report[name] = rep
        print(name, json.dumps(rep, indent=1), flush=True)
    old = json.loads((a.out / "quant_check.json").read_text()) if (a.out / "quant_check.json").exists() else {}
    old.update(report)
    (a.out / "quant_check.json").write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
