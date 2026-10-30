"""GLM v1: build the "general" knowledge pack (Wikidata core of KG16 + the KG16 R0 scorer) as lazily loaded shards.

Inputs (nothing is trained here):
  --src    KG16 pack source (kg16.pack_src output: entities.tsv, relations.tsv, triples.npz, literals.tsv, meta.json)
  --model  KG16 R0 run dir (model.pt + train.json; seed 1 = best R0 seed on Mintaka dev)
  --linker KG16 linker.pt (KG15 14 -> 32 -> 1 MLP); used with KG17's fast candidate form (exact + label prefix)
  --mind   mind repo (for the KG15 code in .claude/worktrees/kg16/shannon/g)

  GLM_WORK=<scratch> python tools/glm_general/build.py --src ... --model ... --linker ... --mind ...

Writes packs/general/ (layout in PACK_FORMAT.md, "shannon-pack/2"):
  manifest.json, weights.bin, node.bin, litval.bin, LICENSE.txt,
  idx/NNNN.txt (name index), names/NNN.txt (labels by id range), adj/NNN.bin, sig/NNN.bin, lit/NNN.bin
"""
import argparse, hashlib, json, math, os, shutil, sys, time, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--linker", required=True)
ap.add_argument("--mind", required=True)
ap.add_argument("--site", default=os.path.dirname(os.path.dirname(HERE)))
ap.add_argument("--adj-bytes", type=int, default=96 * 1024)
ap.add_argument("--sig-bytes", type=int, default=64 * 1024)
ap.add_argument("--lit-bytes", type=int, default=64 * 1024)
ap.add_argument("--idx-shards", type=int, default=2048)
ap.add_argument("--names-per-shard", type=int, default=2048)
a = ap.parse_args()
os.environ["PACK_SRC"] = a.src
core.SRC = a.src
sys.path.insert(0, os.path.join(a.mind, ".claude/worktrees/kg16/shannon/g"))
from kg15 import common as C, encode as EN, link as LK, paths as KP, train as TR  # noqa: E402
import ref  # noqa: E402  (index construction shared with the Python reference)

OUT = os.path.join(a.site, "packs/general")
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for d in ("idx", "names", "adj", "sig", "lit"):
    os.makedirs(os.path.join(OUT, d))
t0 = time.time()
g = core.load()
NE, N = g["NE"], g["N"]
ip, sr, ds = g["indptr"], g["srel"].astype(np.int64), g["dst"].astype(np.int64)
E = len(ds)
print("graph", N, E, flush=True)


# ------------------------------------------------------------------ LEB128 varints (vectorised)
def leb(v):
    v = np.asarray(v, np.uint64)
    nb = np.ones(len(v), np.int64)
    for k in range(1, 10):
        nb += v >= (np.uint64(1) << np.uint64(7 * k))
    out = np.zeros(int(nb.sum()), np.uint8)
    pos = np.r_[0, np.cumsum(nb)[:-1]]
    for k in range(int(nb.max()) if len(v) else 0):
        m = nb > k
        byte = ((v[m] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        byte |= np.where(nb[m] > k + 1, 0x80, 0).astype(np.uint8)
        out[pos[m] + k] = byte
    return out, nb


def shard_ranges(node_bytes, target):
    """contiguous node ranges [lo, hi) with about `target` body bytes each (a big node gets a shard of its own)"""
    los, acc = [0], 0
    for v, b in enumerate(node_bytes.tolist()):
        if acc and acc + b > target:
            los.append(v); acc = 0
        acc += b
    return los


def write_shards(name, vals, node_pos, node_ids_lo, n_nodes, target):
    """vals: flat value array; node_pos[k] = first value index of node (node_ids_lo + k), node_pos[n] = len(vals).
    File: u32 lo | u32 n | u32 off[n+1] (byte offsets into the body) | body (LEB128 values)."""
    enc, nb = leb(vals)
    cb = np.r_[0, np.cumsum(nb)]
    byte_pos = cb[node_pos]
    node_bytes = np.diff(byte_pos)
    los = shard_ranges(node_bytes, target)
    his = los[1:] + [n_nodes]
    total = 0
    gz = 0
    for s, (lo, hi) in enumerate(zip(los, his)):
        off = (byte_pos[lo:hi + 1] - byte_pos[lo]).astype("<u4")
        body = enc[byte_pos[lo]:byte_pos[hi]]
        blob = np.array([lo + node_ids_lo, hi - lo], "<u4").tobytes() + off.tobytes() + body.tobytes()
        open(os.path.join(OUT, name, f"{s:03d}.bin"), "wb").write(blob)
        total += len(blob)
        gz += len(zlib.compress(blob, 6))
    print(f"{name}: {len(los)} shards, {total / 1e6:.2f} MB ({gz / 1e6:.2f} MB gzip)", flush=True)
    return dict(dir=name, lo=[int(x) + node_ids_lo for x in los], n=n_nodes, bytes=int(total), gzip_bytes=int(gz))


# ------------------------------------------------------------------ adjacency + signatures
src = np.repeat(np.arange(N), np.diff(ip))
gstart = np.r_[True, (src[1:] != src[:-1]) | (sr[1:] != sr[:-1])]          # first edge of each (node, srel) group
gs = np.flatnonzero(gstart)                                               # group -> first edge
ge = np.r_[gs[1:], E]
G = len(gs)
gnode = src[gs]
gsrel = sr[gs]
gcnt = ge - gs
first_group = np.searchsorted(gnode, np.arange(N + 1))                    # node -> first group index
ngroups = np.diff(first_group)
prev_srel = np.where(np.r_[True, gnode[1:] != gnode[:-1]], 0, np.r_[0, gsrel[:-1]])
d_srel = gsrel - prev_srel
selfloop = np.zeros(G, np.int64)
sl = np.flatnonzero(ds == src)
if len(sl):
    np.add.at(selfloop, np.searchsorted(gs, sl, side="right") - 1, 1)
print("groups", G, "self loops", len(sl), flush=True)
# member deltas
mdel = ds.copy()
mdel[~gstart] = ds[~gstart] - np.r_[0, ds[:-1]][~gstart]
assert (mdel[~gstart] > 0).all()
# adjacency values: node header at v + 2*first_group[v] + ip[v]; group header at (v+1) + 2j + gs[j]; member e at pg+2+(e-gs)
nv = N + 2 * G + E
vals = np.zeros(nv, np.uint64)
pn = np.arange(N) + 2 * first_group[:N] + ip[:N]
vals[pn] = ngroups
pg = gnode + 1 + 2 * np.arange(G) + gs
vals[pg] = d_srel
vals[pg + 1] = gcnt
eg = np.repeat(np.arange(G), gcnt)
vals[pg[eg] + 2 + (np.arange(E) - gs[eg])] = mdel
adj = write_shards("adj", vals, np.r_[pn, nv], 0, N, a.adj_bytes)
# signatures: node header at v + 2*first_group[v]; group header at v+1+2j: (d_srel, 2*cnt + self)
nv = N + 2 * G
vals = np.zeros(nv, np.uint64)
pn = np.arange(N) + 2 * first_group[:N]
vals[pn] = ngroups
pg = gnode + 1 + 2 * np.arange(G)
vals[pg] = d_srel
vals[pg + 1] = 2 * gcnt + (selfloop > 0)
assert selfloop.max(initial=0) <= 1
sig = write_shards("sig", vals, np.r_[pn, nv], 0, N, a.sig_bytes)
# literal facts per entity: forward edges to literal nodes: n, then (relation id, literal node - NE) pairs
lm = (src < NE) & (sr % 2 == 0) & (ds >= NE)
ls, lr, ld = src[lm], sr[lm] // 2, ds[lm] - NE
lcnt = np.bincount(ls, minlength=NE)
lfirst = np.r_[0, np.cumsum(lcnt)]
nv = NE + 2 * len(ls)
vals = np.zeros(nv, np.uint64)
pn = np.arange(NE) + 2 * lfirst[:NE]
vals[pn] = lcnt
pe = ls + 1 + 2 * np.arange(len(ls))
vals[pe] = lr
vals[pe + 1] = ld
lit = write_shards("lit", vals, np.r_[pn, nv], 0, NE, a.lit_bytes)

# ------------------------------------------------------------------ node table: degree (popularity, for ties and the linker), pack degree
deg, pdeg = g["deg"], g["pdeg"]
b1, _ = leb(deg)
b2, _ = leb(pdeg)
lk = np.array([1 if k == "Q" else 0 for k in g["lkind"][NE:]], np.uint8)
open(os.path.join(OUT, "node.bin"), "wb").write(b1.tobytes() + b2.tobytes() + lk.tobytes())
node_file = dict(path="node.bin", deg_bytes=int(len(b1)), pdeg_bytes=int(len(b2)), lkind_bytes=int(len(lk)))
np.asarray(g["num"][NE:], "<f8").tofile(os.path.join(OUT, "litval.bin"))

# ------------------------------------------------------------------ names (labels by id range; literal nodes: display string)
NPS = a.names_per_shard
names_lo = list(range(0, N, NPS))
for s, lo in enumerate(names_lo):
    with open(os.path.join(OUT, "names", f"{s:03d}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(n.replace("\n", " ").replace("\r", " ") for n in g["names"][lo:lo + NPS]) + "\n")

# ------------------------------------------------------------------ name index (exact aliases + label prefixes), sharded by first word
exact, pref = ref.indexes(g)
S = a.idx_shards


def fnv(s):
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return h


b36 = np.base_repr
lines = {}
for k, v in exact.items():
    lines.setdefault(k, [None, None])[0] = ",".join(b36(int(x), 36).lower() for x in v)
for k, (v, ln) in pref.items():
    lines.setdefault(k, [None, None])[1] = ",".join(f"{b36(int(x), 36).lower()}:{int(l)}" for x, l in zip(v, ln))
first_bytes = {}
for k, (e, p) in lines.items():
    t = k.split(" ")[0]
    first_bytes[t] = first_bytes.get(t, 0) + len(k) + len(e or "") + len(p or "") + 4
SPLIT_AT = 48 * 1024
split = sorted(t for t, b in first_bytes.items() if b > SPLIT_AT)


def shard_key(k):
    t = k.split(" ")
    return t[0] + " " + t[1] if (t[0] in split_set and len(t) > 1) else t[0]


split_set = set(split)
buckets = [[] for _ in range(S)]
for k in sorted(lines):
    e, p = lines[k]
    buckets[fnv(shard_key(k)) % S].append(f"{k}\t{e or ''}\t{p or ''}")
isz = []
for s, b in enumerate(buckets):
    t = ("\n".join(b) + "\n") if b else ""
    open(os.path.join(OUT, "idx", f"{s:04d}.txt"), "w", encoding="utf-8").write(t)
    isz.append(len(t.encode()))
print(f"idx: {S} shards, {sum(isz) / 1e6:.1f} MB, max {max(isz) / 1e3:.0f} kB, split first words {len(split)}", flush=True)

# ------------------------------------------------------------------ weights (KG16 R0 heads, relation operators, linker)
seed = json.load(open(os.path.join(a.model, "train.json")))["seed"]
m = TR.Model(seed)
T, M = EN.tokens(g["rel_name"], "cpu")
m.s1.set_rel_tokens(torch.tensor(T), torch.tensor(M))
m.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location="cpu"))
m.eval()
with torch.no_grad():
    Lrel = m.s1.rel_logs(torch.zeros(len(g["rel_name"]), 384)).float()     # (R, 16, 16) = fnorm(read(name tokens)), ||.||_F = 4
lkr = LK.Linker()
lkr.load_state_dict(torch.load(a.linker, map_location="cpu"))
sd, ld_ = m.state_dict(), lkr.state_dict()
parts = [("mu", sd["s1.mu"]), ("sd", sd["s1.sd"]), ("Wq", sd["s1.Wq.weight"]), ("bq", sd["s1.Wq.bias"]),
         ("wa", sd["s1.w.weight"].reshape(-1)), ("ba", sd["s1.w.bias"]), ("logtau", sd["s1.logtau"].reshape(1)),
         ("beta", sd["beta"].reshape(1)), ("h1_W", sd["h1.weight"]), ("h1_b", sd["h1.bias"]),
         ("dir_W", sd["dir.weight"].reshape(-1)), ("dir_b", sd["dir.bias"]), ("L", Lrel),
         ("link_W1", ld_["f.0.weight"]), ("link_b1", ld_["f.0.bias"]), ("link_W2", ld_["f.2.weight"].reshape(-1)),
         ("link_b2", ld_["f.2.bias"])]
layout, off, blob = {}, 0, []
for k, v in parts:
    v = v.detach().float().contiguous().numpy()
    layout[k] = {"offset": off, "shape": list(v.shape)}
    off += v.size
    blob.append(v.ravel())
wts = np.concatenate(blob).astype("<f4")
wts.tofile(os.path.join(OUT, "weights.bin"))
params = dict(s1=int(sum(p.numel() for p in m.s1.parameters())), beta=1, h1=6 * 384 + 6, dir=385, linker=lkr.nparams())

# ------------------------------------------------------------------ relations, licence, manifest
PREP = (" of", " in", " by", " for", " at", " on", " to", " from", " with", " as")
rels = []
for pid, nm in zip(g["rel_pid"], g["rel_name"]):
    inv = ("← " + nm) if (nm.endswith(PREP) or nm.startswith(("is ", "has "))) else nm + " of"
    rels.append({"id": pid, "label": nm, "inverse_label": inv})
meta = json.load(open(os.path.join(a.src, "meta.json")))
LIC = """General knowledge pack (Shannon GLM v1)

Contents: a 303,149-entity core of Wikidata (labels, English aliases, 1,089 relations, 3,462,612 entity facts and
1,044,490 date / quantity facts) as used by Shannon's KG15/KG16 experiments, plus trained scorer weights.

Facts, labels and aliases come from Wikidata (https://www.wikidata.org), whose structured data is released under
Creative Commons CC0 1.0 (public domain dedication): https://creativecommons.org/publicdomain/zero/1.0/
The entity facts were selected through Wikidata5M (Wang et al., 2021, "KEPLER: A Unified Model for Knowledge Embedding
and Pre-trained Language Representation", TACL; https://deepgraphlearning.github.io/project/wikidata5m). Wikidata5M is a
derived dataset built from Wikidata and English Wikipedia; its distribution does not state a licence of its own. This
pack ships only Wikidata facts (entity ids, relation ids, labels, aliases, dates and quantities) and no Wikipedia text,
so the CC0 terms of the underlying Wikidata data apply to its contents. Date and quantity facts and the English labels
were fetched from the Wikidata Query Service on 2026-10-25.

Selection: the top 300,000 entities by Wikidata5M degree plus the entities annotated in the Mintaka train and dev
questions (Mintaka: Sen et al., 2022, CC BY 4.0; no Mintaka question text is included in this pack). Facts are kept
only when both ends are in the core.

Scorer and linker weights: trained by the Shannon project on Mintaka train questions (KG16 R0, seed 1); released with
this demo under the same terms as the rest of the repository.
Encoder (not in this pack): sentence-transformers/all-MiniLM-L6-v2, Apache-2.0.
"""
open(os.path.join(OUT, "LICENSE.txt"), "w").write(LIC)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tree_bytes(d):
    tot = gz = 0
    for f in sorted(os.listdir(os.path.join(OUT, d))):
        b = open(os.path.join(OUT, d, f), "rb").read()
        tot += len(b); gz += len(zlib.compress(b, 6))
    return tot, gz


manifest = {
    "format": "shannon-pack/2",
    "id": "general",
    "kind": "wikidata-sharded",
    "title": "General knowledge (Wikidata)",
    "description": "A 303k-entity core of Wikidata: people, places, works, organisations, with dates and quantities.",
    "license": "CC0 1.0 (Wikidata data); see LICENSE.txt",
    "attribution": "Facts and labels from Wikidata (CC0), selected through Wikidata5M (Wang et al., 2021).",
    "relations": rels,
    "inverse_suffix": "^-1",
    "counts": {"entities": NE, "literal_nodes": N - NE, "nodes": N, "entity_facts": meta["triples"],
               "literal_facts": meta["literal_facts"], "directed_edges": int(E), "groups": int(G)},
    "cut": {"from": "KG16 graph: 4,818,335 entities, 20,624,575 entity triples (Wikidata5M) + 1,044,490 literal facts",
            "kept": "entities in the top 300,000 by Wikidata5M degree + Mintaka train/dev annotated entities; facts with both ends kept",
            "dropped": "4,515,186 lower-degree entities and every fact touching them (17.16M of 20.62M entity triples)"},
    "files": {
        "weights": {"path": "weights.bin", "bytes": int(wts.nbytes), "sha256": sha(os.path.join(OUT, "weights.bin")),
                    "dtype": "f32le", "layout": layout},
        "node": dict(node_file, bytes=os.path.getsize(os.path.join(OUT, "node.bin")), sha256=sha(os.path.join(OUT, "node.bin"))),
        "litval": {"path": "litval.bin", "bytes": os.path.getsize(os.path.join(OUT, "litval.bin")), "dtype": "f64le",
                   "sha256": sha(os.path.join(OUT, "litval.bin"))},
        "adj": adj, "sig": sig, "lit": lit,
        "names": {"dir": "names", "per_shard": NPS, "shards": len(names_lo), "bytes": tree_bytes("names")[0]},
        "idx": {"dir": "idx", "shards": S, "hash": "fnv1a32(shard key) % shards", "split_first_words": split,
                "bytes": int(sum(isz))},
    },
    "linker": {"type": "kg17-fast-v1", "maxn": LK.MAXN, "keycap": LK.KEYCAP, "prefix_new": 8, "top_topics": 3,
               "features": ["exact", "cos", "n/5", "len/30", "log1p(deg)/10", "maximal", "log1p(amb)/5", "1/(1+rank)",
                            "is_label", "pos", "ctx", "ctx*exact", "cos*(1-exact)", "1"],
               "source": "KG16 linker.pt (KG15 MLP, trained on Mintaka train) with KG17's exact + label-prefix candidates"},
    "scorer": {"type": "s1-sym-v1", "encoder": "all-MiniLM-L6-v2", "tok_dim": 384, "max_tokens": 32, "sent_max_tokens": 64,
               "dim": 16, "prod": "plain", "max_hops": 2, "cand_cap": KP.CAP, "hop2_max": KP.HOP2_MAX,
               "arg_max_set": KP.ARG_MAX_S, "ops": KP.OPS, "tok_fp16": True,
               "source": "KG16 R0 (frozen MiniLM, KG15 S1-sym + H1 + direction heads), seed 1 (best R0 seed on Mintaka dev, 23.1)",
               "params_trained": params, "params_total_trained": int(sum(params.values()))},
    "eval": {},
}
sizes = {}
for d in ("adj", "sig", "lit", "names", "idx"):
    sizes[d] = tree_bytes(d)
for f in ("weights.bin", "node.bin", "litval.bin"):
    b = open(os.path.join(OUT, f), "rb").read()
    sizes[f] = (len(b), len(zlib.compress(b, 6)))
manifest["sizes"] = {k: {"bytes": v[0], "gzip": v[1]} for k, v in sizes.items()}
manifest["sizes"]["total"] = {"bytes": sum(v[0] for v in sizes.values()), "gzip": sum(v[1] for v in sizes.values())}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(json.dumps(manifest["sizes"], indent=1))
print("params", params, "built in", round(time.time() - t0), "s")
