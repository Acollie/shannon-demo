"""GLM v1 Python reference: the KG16 R0 pipeline (KG15 code, unchanged) run on the general pack's core graph with the
KG17-style fast linker (exact alias index + label-prefix index, no dense search). Inference only.
  MIND=<mind repo> KG16_MODEL=<R0/s1> KG16_LINKER=<linker.pt> PACK_SRC=<kg16_pack_src> GLM_WORK=<scratch> \
    python tools/glm_general/ref.py --qfile mintaka_dev.json --n 200 --out tools/glm_parity/ref_general.json
"""
import argparse, json, math, multiprocessing as mp, os, sys, time
import numpy as np, torch
MIND = os.environ.get("MIND", "/Users/alexcollie/Documents/code/github/mind")
sys.path.insert(0, os.path.join(MIND, ".claude/worktrees/kg16/shannon/g"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg15 import common as C, encode as EN, link as LK, paths as KP, train as TR  # noqa
import core
torch.set_num_threads(8)
MODEL = os.environ.get("KG16_MODEL", "")          # KG16 R0 run dir (seed 1)
LINKER = os.environ.get("KG16_LINKER", "")        # KG16 linker.pt
STOP = {"the", "a", "an", "of", "in", "on", "and", "for", "to", "at", "by", "with", "from", "is", "who", "what"}
HOP2_MAX = int(os.environ.get("HOP2_MAX", KP.HOP2_MAX))
KP.HOP2_MAX = HOP2_MAX


def graph():
    g = core.load()
    z = dict(indptr=g["indptr"], srel=g["srel"], dst=g["dst"], kind=g["kind"], num=g["num"], lkind=g["lkind"], deg=g["deg"])
    G = KP.Graph(z, dict(names=g["names"], qids=g["qids"], NE=g["NE"]))
    return g, G


def indexes(g):
    """exact: norm(label | alias) -> ids (<= 50, by degree desc, then id); prefix: KG17 label-prefix index"""
    import collections
    deg, pdeg, NE = g["deg"], g["pdeg"], g["NE"]
    ex, pr = collections.defaultdict(list), collections.defaultdict(list)
    for i in range(NE):
        if pdeg[i] == 0:
            continue
        ks = {C.norm(g["names"][i])} | {C.norm(a) for a in g["aliases"][i]}
        for k in sorted(ks):
            if k:
                ex[k].append(i)
        t = C.norm(g["names"][i]).split()
        if t and t[0] == "the" and len(t) > 1:
            pr[" ".join(t[1:])].append(i)
        for n in range(1, len(t)):
            if n == 1 and (t[0] in STOP or len(t[0]) < 4):
                continue
            pr[" ".join(t[:n])].append(i)
    def top(v):
        v = np.array(v, np.int64)
        return v[np.argsort(-deg[v], kind="stable")[:LK.KEYCAP]]
    exact = {k: top(v).astype(np.int32) for k, v in ex.items()}
    pref = {}
    for k, v in pr.items():
        v = top(v)
        pref[k] = (v.astype(np.int32), np.array([len(C.norm(g["names"][c]).split()) for c in v], np.int16))
    return exact, pref


def features(g, exact, pref, questions, qvec, name_vec, kp=8):
    """KG17 features_fast, verbatim logic (exact channel + up to kp prefix entities)"""
    deg, names = g["deg"], g["names"]
    rows = []
    for qi, q in enumerate(questions):
        toks = C.norm(q).split()
        best = {}
        hits = [(i, n, s) for i, n, s in LK.ngrams(toks, LK.MAXN) if s in exact]
        spans = [(i, i + n) for i, n, _ in hits]
        L = max(len(toks), 1)
        for i, n, s in hits:
            maximal = not any(a <= i and i + n <= b and (b - a) > n for a, b in spans)
            ids = exact[s]
            for rk, c in enumerate(ids):
                c = int(c)
                r = (1.0, 1.0, n, len(s), maximal, len(ids), 1.0 / (1 + rk), float(C.norm(names[c]) == s), i / L)
                if c not in best or r[:4] > best[c][:4]:
                    best[c] = r
        ph = sorted([(n, i, s) for i, n, s in LK.ngrams(toks, LK.MAXN) if s in pref], reverse=True)
        added = 0
        for n, i, s in ph:
            ids, ln = pref[s]
            for rk, (c, lt) in enumerate(zip(ids.tolist(), ln.tolist())):
                if c in best:
                    continue
                best[c] = (0.0, n / max(lt, 1), n, len(s), True, len(ids), 1.0 / (1 + rk), 0.0, i / L)
                added += 1
                if added >= kp:
                    break
            if added >= kp:
                break
        rows.append(best)
    allc = sorted({c for b in rows for c in b})
    NV = name_vec([names[c] for c in allc])
    pos = {c: k for k, c in enumerate(allc)}
    out = []
    for qi, best in enumerate(rows):
        nodes = np.array(sorted(best), np.int64)
        X = np.zeros((len(nodes), LK.FEAT), np.float32)
        for k, c in enumerate(nodes):
            ex_, cs, n, ln, mx, amb, rk, isl, p = best[c]
            ctx = float(qvec[qi] @ NV[pos[c]])
            X[k] = [ex_, cs, n / 5, ln / 30, math.log1p(deg[c]) / 10, float(mx), math.log1p(amb) / 5, rk, isl, p, ctx,
                    ctx * ex_, cs * (1 - ex_), 1.0]
        out.append((nodes, X))
    return out


G = None
def _cand(topics):
    return KP.question(G, topics, None, np.zeros(G.N, bool))


def model_and_rels(g):
    seed = json.load(open(os.path.join(MODEL, "train.json")))["seed"]
    m = TR.Model(seed)
    T, M = EN.tokens(g["rel_name"], "cpu")
    m.s1.set_rel_tokens(torch.tensor(T), torch.tensor(M))
    m.load_state_dict(torch.load(os.path.join(MODEL, "model.pt"), map_location="cpu"))
    m.eval()
    E = torch.zeros(len(g["rel_name"]), 384)
    return m, E


def main():
    global G
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qfile", required=True, help="mintaka_dev.json (dev only; never test)")
    a = ap.parse_args()
    t0 = time.time()
    g, G = graph()
    exact, pref = indexes(g)
    print("indexes", len(exact), len(pref), round(time.time() - t0, 1), flush=True)
    qs = json.load(open(a.qfile))[:a.n]
    texts = [q["question"] for q in qs]
    t1 = time.time()
    qvec = EN.sent(texts, "cpu")
    feats = features(g, exact, pref, texts, qvec, lambda s: EN.sent(s, "cpu", bs=512))
    lk = LK.Linker(); lk.load_state_dict(torch.load(LINKER, map_location="cpu")); lk.eval()
    links = LK.apply(lk, feats, "cpu")
    t_link = time.time() - t1
    t1 = time.time()
    with mp.get_context("fork").Pool(10) as pool:
        cands = pool.map(_cand, [L["topics"] for L in links], chunksize=2)
    t_list = time.time() - t1
    t1 = time.time()
    T, M = EN.tokens(texts, "cpu")
    e = dict(T=T, M=M, S=EN.sent(texts, "cpu"))
    model, E = model_and_rels(g)
    preds = TR.predict(model, G, dict(cands=cands, links=links), e, E, dev="cpu", keep_top=8)
    t_score = time.time() - t1
    hits = TR.score(qs, [p["answer"] for p in preds])
    rows = []
    for q, L, c, p, h, (nodes, X) in zip(qs, links, cands, preds, hits, feats):
        rows.append(dict(id=q["id"], q=q["question"], type=q["complexityType"], topics=[g["qids"][t] for t in L["topics"]],
                         topic_ids=L["topics"], tlog=L["tlog"], n_link_cands=len(nodes), link_nodes=nodes.tolist(), n_cands=len(c["size"]), n_full=c["n_full"],
                         op=p["op"], how=p["how"], top=p["topk"][:3], answer=p["answer"], node=p["node"], text=p["text"], hit=h))
    acc = float(np.mean(hits))
    by = TR.by_type(qs, hits)
    rel = LK.recall_at(dict(qids=g["qids"], _eid={q: i for i, q in enumerate(g["qids"])}), links, qs, g["NE"])
    summ = dict(n=len(qs), hits1=acc, by_type=by, link_recall=rel, hop2_max=HOP2_MAX, model=MODEL,
                secs=dict(link=t_link, list=t_list, score=t_score, total=time.time() - t0))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
