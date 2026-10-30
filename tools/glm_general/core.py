"""GLM v1 general pack: the KG16 core (pack source) as a KG15-style CSR graph, node ids renumbered by degree.
Entities 0..NE-1 sorted by Wikidata5M-graph degree (desc), then literal value nodes NE..N-1 by pack degree (desc).
Edges: entity triples both ways (2r, 2r+1) and literal facts both ways, deduplicated, sorted by (src, srel, dst)."""
import numpy as np, os, pickle, time
SRC = os.environ.get("PACK_SRC", "/private/tmp/claude-501/-Users-alexcollie-Documents-code-github-mind/9638aa00-505b-42d1-b96e-c8e770011c2b/scratchpad/kg16_pack_src")
HERE = os.environ.get("GLM_WORK", os.path.dirname(os.path.abspath(__file__)))


def csr(src, srel, dst, N):
    o = np.lexsort((dst, srel, src)); src, srel, dst = src[o], srel[o], dst[o]
    keep = np.r_[True, (src[1:] != src[:-1]) | (srel[1:] != srel[:-1]) | (dst[1:] != dst[:-1])]
    src, srel, dst = src[keep], srel[keep], dst[keep]
    indptr = np.zeros(N + 1, np.int64); np.cumsum(np.bincount(src, minlength=N), out=indptr[1:])
    return indptr, srel, dst


def load(cache=os.path.join(HERE, "core.pkl")):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    t0 = time.time()
    qids, names, deg, aliases = [], [], [], []
    for line in open(os.path.join(SRC, "entities.tsv"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        qids.append(p[0]); names.append(p[1]); deg.append(int(p[2])); aliases.append([a for a in p[3].split("|") if a] if len(p) > 3 else [])
    rel_pid, rel_name = [], []
    for line in open(os.path.join(SRC, "relations.tsv"), encoding="utf-8"):
        i, pid, nm = line.rstrip("\n").split("\t"); rel_pid.append(pid); rel_name.append(nm)
    pidx = {p: i for i, p in enumerate(rel_pid)}
    eid = {q: i for i, q in enumerate(qids)}
    NE = len(qids)
    z = np.load(os.path.join(SRC, "triples.npz"))
    H, R, T = z["h"].astype(np.int64), z["r"].astype(np.int64), z["t"].astype(np.int64)
    lit, lnames, lnum, lk = {}, [], [], []
    S, Rl, D = [], [], []
    for line in open(os.path.join(SRC, "literals.tsv"), encoding="utf-8"):
        q, pid, disp, k, num = line.rstrip("\n").split("\t")
        key = (k, disp)
        if key not in lit:
            lit[key] = NE + len(lnames); lnames.append(disp); lnum.append(float(num)); lk.append(k)
        S.append(eid[q]); Rl.append(pidx[pid]); D.append(lit[key])
    S, Rl, D = np.array(S, np.int64), np.array(Rl, np.int64), np.array(D, np.int64)
    src = np.concatenate([H, T, S, D]); srel = np.concatenate([2 * R, 2 * R + 1, 2 * Rl, 2 * Rl + 1]); dst = np.concatenate([T, H, D, S])
    N = NE + len(lnames)
    indptr, srel, dst = csr(src, srel, dst, N)
    pdeg = np.diff(indptr)
    deg = np.array(deg, np.int64)
    # renumber: entities by full degree desc, literals by pack degree desc (stable)
    eo = np.lexsort((np.arange(NE), -deg))
    lo = NE + np.lexsort((np.arange(N - NE), -pdeg[NE:]))
    order = np.r_[eo, lo]                      # new id -> old id
    new = np.empty(N, np.int64); new[order] = np.arange(N)
    src = np.repeat(np.arange(N), pdeg)
    indptr, srel, dst = csr(new[src], srel, new[dst], N)
    pdeg2 = np.diff(indptr)
    allnames = names + lnames
    g = dict(qids=[qids[i] for i in eo], names=[allnames[i] for i in order], aliases=[aliases[i] for i in eo], NE=NE, N=N,
             rel_pid=rel_pid, rel_name=rel_name, indptr=indptr, srel=srel.astype(np.int32), dst=dst.astype(np.int32),
             kind=np.r_[np.zeros(NE, np.int8), np.ones(N - NE, np.int8)],
             num=np.r_[np.full(NE, np.nan), np.array(lnum)[lo - NE]],
             lkind=np.array([""] * NE + [lk[i - NE] for i in lo], dtype="<U1"),
             deg=np.r_[deg[eo], pdeg2[NE:]], pdeg=pdeg2)
    pickle.dump(g, open(cache, "wb"), protocol=4)
    print("built", N, len(dst), round(time.time() - t0, 1), "s")
    return g


if __name__ == "__main__":
    g = load()
    print(g["NE"], g["N"], len(g["dst"]), g["names"][:5], g["names"][g["NE"]:g["NE"] + 5])
