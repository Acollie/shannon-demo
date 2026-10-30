"""Run treaty-graph SystemX (unchanged) on the demo's example docs/questions and dump every intermediate.

cd treaty-graph && nice -n 19 uv run python <this> <examples.json> <docs_dir> <out.json> [--extra extra.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from treatygraph import generic  # noqa: E402

ex = json.loads(Path(sys.argv[1]).read_text())
docs_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
docs = ex["docs"]
if len(sys.argv) > 5 and sys.argv[4] == "--extra":
    docs = docs + json.loads(Path(sys.argv[5]).read_text())["docs"]

res = []
sysx = None
for d in docs:
    text = (docs_dir / Path(d["file"]).name).read_text()
    if sysx is None:
        sysx = generic.SystemX(text)
        sysx.ev.yn.dev = 'cpu'; sysx.ev.yn.m.to('cpu')   # CPU, deterministic (YesNo would pick MPS)
    else:
        sysx.ev = sysx.ev.reset(text)
    ev = sysx.ev
    drec = {"id": d["id"], "passages": [[o, p] for o, p in ev.pars], "questions": []}
    for q in d["questions"]:
        q = q["q"]
        t0 = time.time()
        picked, cos, ovl, top_e, top_o, qv, qw = ev.retrieve(q)
        cs = ev.candidates(q)
        lpf = sysx.fm.logp(ev.enc([q])[0])
        rows = generic.featurize(cs, lpf)
        x = sysx.gate._mat(rows)
        x = (x - sysx.gate.mu) / sysx.gate.sd * (x != 0)
        s = x @ sysx.gate.w
        i = int(np.argmax(s))
        srt = np.sort(s)[::-1]
        ans, cite, c = sysx.ask(q)
        assert c is cs[i] or (c.form == cs[i].form and c.answer == cs[i].answer)
        # tie diagnostics for numpy argsort
        def ties(a, k=3):
            o = np.sort(a)[::-1]
            return bool(len(o) > k and o[k - 1] == o[k])
        drec["questions"].append({
            "q": q, "answer": ans, "form": c.form, "pos": c.pos, "margin": float(srt[0] - srt[1]),
            "chosen": i, "picked": [int(p) for p in picked], "by_e": [int(p) for p in (-cos).argsort()[:3]],
            "by_o": [int(p) for p in (-ovl).argsort()[:3]], "ovl_tie": ties(ovl), "cos_tie": ties(cos),
            "cos": cos.tolist(), "ovl": ovl.tolist(), "lpf": lpf.tolist(),
            "cands": [{"form": c.form, "answer": c.answer, "pos": c.pos, "feats": c.feats, "score": float(sc)}
                      for c, sc in zip(cs, s)],
            "secs": time.time() - t0})
        print(d["id"], q, "->", c.form, ans, f"{srt[0] - srt[1]:.2f}", flush=True)
    res.append(drec)
out_path.write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("->", out_path)
