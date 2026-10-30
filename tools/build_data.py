"""Derived data for the site, from a Shannon checkpoint and PUBLIC held-out FineWeb-Edu windows.

  python tools/build_data.py --shannon-src /path/to/shannon --ckpt runs/X/cooldown_1.0.pt --windows heldout.npz \
      --vocab vocab.txt --out models/<model-id>

Writes into --out:
  dictionary.json   per code: usage count, top source spans, contexts, distinctive wordpieces, usage by idea slot
  map.json          2-D UMAP of the 4,096 codebook vectors (cosine), nearest neighbours, k-means region labels
  presets.json      short held-out snippets (decoded, lower-cased) with their true continuation
  stats.json        held-out BPB / G, codebook usage (rank-frequency), modal ("null") code
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


STOP = set("""about above after again against also among because been before being below between both but could does doing down during each
even ever every from further have having here hers herself himself however into itself just like made make many more most much must
never only other ours over same should since some such than that their theirs them then there these they this those through under until
very were what when where which while whom will with within without would your yours also often well used using known""".split())


class Vocab:
    def __init__(self, path):
        self.toks = [l.rstrip("\n") for l in open(path, encoding="utf-8")]

    def decode(self, ids):
        out = []
        for i in ids:
            p = self.toks[int(i)]
            if p.startswith("##") and out:
                out[-1] += p[2:]
            else:
                out.append(p)
        s = " ".join(out)
        s = re.sub(r" ([.,;:!?%)\]}])", r"\1", s)
        s = re.sub(r"([(\[{$]) ", r"\1", s)
        s = re.sub(r" ' (s|t|re|ve|ll|d|m)\b", r"'\1", s)
        s = re.sub(r" - ", "-", s)
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shannon-src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--presets", default="", help="comma-separated idx:Title pairs of held-out windows to use as presets (default: auto)")
    ap.add_argument("--n-regions", type=int, default=28)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.shannon_src))
    from models import ModelConfig, build_model, shift_right

    rc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.json")))
    cfg = ModelConfig.from_dict(rc["cfg"])
    m = build_model(cfg)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False))
    m.eval()
    voc = Vocab(a.vocab)
    W = np.load(a.windows)
    src = torch.from_numpy(W["src"].astype(np.int64))
    tgt = torch.from_numpy(W["tgt"].astype(np.int64))
    tb = W["tb"].astype(np.float64)
    N, M, R, K = len(src), cfg.n_ideas, cfg.R, cfg.K

    # ---- codes + BPB (true / shuffled = previous window's ideas / null = modal idea everywhere)
    codes, nll = [], {"true": [], "shuffle": [], "null": []}
    with torch.no_grad():
        for i in range(0, N, 64):
            codes.append(m.encode_to_codes(src[i:i + 64]))
        codes = torch.cat(codes)
        modal = int(torch.bincount(codes.reshape(-1), minlength=K).argmax())
        partner = (torch.arange(N) - 1) % N
        for v in nll:
            for i in range(0, N, 64):
                j = torch.arange(i, min(N, i + 64))
                c = {"true": codes[j], "shuffle": codes[partner[j]], "null": torch.full_like(codes[j], modal)}[v]
                lg = m.decode_from_codes(c, shift_right(tgt[j]))
                nll[v].append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[j].reshape(-1), reduction="none").view(len(j), -1).sum(1))
    nll = {k: torch.cat(v).numpy() for k, v in nll.items()}
    bpb = {k: float(v.sum() / LN2 / tb.sum()) for k, v in nll.items()}
    print("held-out BPB", bpb)

    cn = codes.numpy()
    flat = cn.reshape(-1)
    counts = np.bincount(flat, minlength=K)
    p = counts / counts.sum()
    ppl = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    srcn = src.numpy()
    spans = srcn.reshape(N, M, R).reshape(-1, R)
    pos_of = np.tile(np.arange(M), N)
    win_of = np.repeat(np.arange(N), M)
    gtok = np.bincount(spans.reshape(-1), minlength=cfg.vocab).astype(np.float64)
    gp = gtok / gtok.sum()
    rs = np.random.RandomState(0)
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(K + 1))

    dictionary = []
    lift_words = {}
    for c in range(K):
        inst = order[bounds[c]:bounds[c + 1]]
        if len(inst) == 0:
            dictionary.append({"n": 0})
            continue
        sp = Counter(tuple(spans[i]) for i in inst)
        tc = Counter(spans[inst].reshape(-1).tolist())
        lift = sorted((((n / (R * len(inst))) / gp[t]), t, n) for t, n in tc.items()
                      if n >= 2 and voc.toks[t].isalpha() and len(voc.toks[t]) > 2 and voc.toks[t] not in STOP and voc.toks[t] not in ("the", "and", "for", "are", "was", "with")
                      and (n / (R * len(inst))) / gp[t] > 3)[::-1][:5]
        lift_words[c] = [(t, n) for _, t, n in lift]
        ctx = []
        for i in rs.choice(inst, min(3, len(inst)), replace=False):
            w, pos = int(win_of[i]), int(pos_of[i])
            lo, hi = pos * R, (pos + 1) * R
            ctx.append([voc.decode(srcn[w, max(0, lo - 6):lo]), voc.decode(srcn[w, lo:hi]), voc.decode(srcn[w, hi:hi + 6])])
        ph = np.bincount(pos_of[inst] // 4, minlength=M // 4).tolist()      # usage by idea slot, 8 bins of 4 slots
        dictionary.append({"n": int(len(inst)), "s": [[voc.decode(list(s)), int(k)] for s, k in sp.most_common(5)],
                           "w": [voc.decode([t]) for t, _ in lift_words[c]], "x": ctx, "p": ph})

    # ---- map: UMAP of the l2-normalised codebook
    cb = F.normalize(m.vq.codebook.float(), dim=-1).numpy()
    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=0).fit_transform(cb)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    sim = cb @ cb.T
    np.fill_diagonal(sim, -2)
    nbr = np.argsort(-sim, axis=1)[:, :8]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=a.n_regions, n_init=4, random_state=0).fit(xy)
    regions = []
    for r in range(a.n_regions):
        mem = np.where(km.labels_ == r)[0]
        wc = Counter()
        for c in mem:
            for t, n in lift_words.get(int(c), []):
                wc[voc.toks[t]] += n
        top = [w for w, _ in wc.most_common(3)]
        inst = np.concatenate([order[bounds[c]:bounds[c + 1]] for c in mem])
        rc_ = Counter(spans[inst].reshape(-1).tolist())
        tot = sum(rc_.values())
        cand = sorted((((n / tot) / gp[t]), t) for t, n in rc_.items()
                      if n >= 6 and voc.toks[t].isalpha() and len(voc.toks[t]) > 3 and voc.toks[t] not in STOP)[::-1]
        top = [voc.toks[t] for _, t in cand[:3]] or top
        regions.append({"xy": [round(float(v), 4) for v in km.cluster_centers_[r]], "label": " · ".join(top), "n": int(len(mem))})
    mp = {"xy": [[round(float(x), 4), round(float(y), 4)] for x, y in xy], "nbr": nbr.tolist(),
          "region": km.labels_.tolist(), "regions": regions}

    # ---- presets
    if a.presets:
        pick = [int(x.split(":")[0]) for x in a.presets.split(",")]
        titles = {int(x.split(":")[0]): x.split(":", 1)[1] for x in a.presets.split(",") if ":" in x}
    else:
        pick, titles = [], {}
        for i in np.argsort(nll["true"] / LN2 / tb.sum(1)):
            t = voc.decode(srcn[i])
            if "[UNK]" in t or len(set(srcn[i])) < 70:
                continue
            pick.append(int(i))
            if len(pick) >= 40:
                break
    presets = [{"i": i, "title": titles.get(i, voc.decode(srcn[i][:6])), "text": voc.decode(srcn[i]), "next": voc.decode(tgt[i].numpy()),
                "bpb": float(nll["true"][i] / LN2 / tb[i].sum())} for i in pick]

    ranks = np.sort(counts)[::-1]
    stats = {"heldout_windows": N, "bpb": bpb, "G_shuffle": bpb["shuffle"] - bpb["true"], "G_null": bpb["null"] - bpb["true"],
             "modal_code": modal, "codes_used": int((counts > 0).sum()), "code_perplexity": ppl, "K": K,
             "rank_counts": ranks.tolist(), "n_instances": int(counts.sum())}
    os.makedirs(a.out, exist_ok=True)
    for name, obj in (("dictionary", dictionary), ("map", mp), ("presets", presets), ("stats", stats)):
        json.dump(obj, open(os.path.join(a.out, f"{name}.json"), "w"), separators=(",", ":"), ensure_ascii=False)
    print({k: v for k, v in stats.items() if k != "rank_counts"})


if __name__ == "__main__":
    main()
