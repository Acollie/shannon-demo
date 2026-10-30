"""GLM v0: build the "movies" pack (MetaQA / WikiMovies graph + the KG3b R2 operator path scorer) and the Python
reference answers used for the JS parity check.

The scorer is KG3b R2 = KG10B "C0" = KG3c "ctl", seed 0 (trained on MetaQA 1+2-hop train questions only, 9,620 params;
KG3c held4 hits@1 seed 0: 1-hop 0.969, 2-hop 1.000, 3-hop 0.780). Nothing is trained here.

Run (Mac, CPU, a few seconds plus MiniLM on ~30 questions):
  PYTHONPATH=<dir with scipy> <mind>/shannon/.venv/bin/python tools/glm_build.py \
      --mind <mind repo> --model <kg3c_models>/ctl_s0/model.pt --onnx <Xenova all-MiniLM-L6-v2>/onnx/model_fp16.onnx --onnx-small <...>/onnx/model_quantized.onnx

Writes:
  packs/movies/{manifest.json, entities.txt, graph.bin, weights.bin, LICENSE.txt}
  models/glm/minilm_{fp16,q8}.partN (MiniLM ONNX split into < 15 MB parts) + models/glm/vocab.txt
  tools/glm_parity/ref.json        (Python pipeline outputs on 20 MetaQA TRAIN questions + the 6 page examples)
"""
import argparse, hashlib, json, os, random, shutil, sys
if len(sys.argv) > 1 and sys.argv[1] == "general":   # GLM v1 general (Wikidata) pack: tools/glm_general/build.py
    import runpy
    sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm_general", "build.py")] + sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
    sys.exit(0)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mind", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--onnx", required=True, help="model_fp16.onnx")
ap.add_argument("--onnx-small", required=True, help="model_quantized.onnx")
ap.add_argument("--vocab", default=None)
ap.add_argument("--site", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.mind, ".claude/worktrees/kg3c/shannon/g"))
from kg1 import kgdata as K                      # noqa: E402
from kg1.kg3b import PATHS, NP, candidates, answer  # noqa: E402
from kg1.kg3c import Composer3                   # noqa: E402
from kg1.walker2 import Env                      # noqa: E402

DATA = os.path.join(a.mind, "shannon/data/metaqa")
PACK = os.path.join(a.site, "packs/movies")
os.makedirs(PACK, exist_ok=True)

# ------------------------------------------------------------------ graph
kb = K.KB(os.path.join(DATA, "kb/kb.txt"))
with open(os.path.join(PACK, "entities.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(kb.names) + "\n")
tr = kb.triples  # sorted unique (h, r, t)
with open(os.path.join(PACK, "graph.bin"), "wb") as f:
    f.write(tr[:, 0].astype("<u4").tobytes()); f.write(tr[:, 2].astype("<u4").tobytes()); f.write(tr[:, 1].astype("u1").tobytes())
shutil.copy(os.path.join(DATA, "LICENSE.txt"), os.path.join(PACK, "LICENSE.txt"))

# ------------------------------------------------------------------ scorer weights
sd = torch.load(a.model, map_location="cpu")
model = Composer3("softmax", 0.0); model.load_state_dict(sd); model.eval()
with torch.no_grad():
    Le = model.edge_logs().double()
    Me = torch.linalg.matrix_exp(Le)
parts = [("mu", sd["mu"]), ("sd", sd["sd"]), ("W", sd["lin.weight"]), ("b", sd["lin.bias"]),
         ("Le", Le.float()), ("Me", Me.float())]
layout, off, blob = {}, 0, []
for k, v in parts:
    v = v.detach().float().contiguous().numpy()
    layout[k] = {"offset": off, "shape": list(v.shape)}
    off += v.size; blob.append(v.ravel())
w = np.concatenate(blob).astype("<f4")
w.tofile(os.path.join(PACK, "weights.bin"))


LAB = {"directed_by": ("directed by", "director of"), "written_by": ("written by", "writer of"),
       "starred_actors": ("starring", "acted in"), "release_year": ("released in", "year of"),
       "in_language": ("in language", "language of"), "has_tags": ("tagged", "tag of"), "has_genre": ("genre", "genre of"),
       "has_imdb_votes": ("IMDb votes", "votes of"), "has_imdb_rating": ("IMDb rating", "rating of")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


manifest = {
    "format": "shannon-pack/1",
    "id": "movies",
    "title": "Movies (MetaQA / WikiMovies)",
    "description": "Films with their directors, writers, actors, years, languages, genres, tags, IMDb votes and ratings.",
    "license": "CC BY 3.0",
    "attribution": "MetaQA (Zhang et al., 2018, 'Variational Reasoning for Question Answering with Knowledge Graph'), "
                   "built on the WikiMovies knowledge base (Miller et al., 2016). CC BY 3.0; see LICENSE.txt.",
    "relations": [{"id": r, "label": LAB[r][0], "inverse_label": LAB[r][1]} for r in K.RELS],
    "inverse_suffix": "^-1",
    "counts": {"entities": kb.n, "triples": int(len(tr))},
    "files": {
        "entities": {"path": "entities.txt", "bytes": os.path.getsize(os.path.join(PACK, "entities.txt"))},
        "graph": {"path": "graph.bin", "bytes": os.path.getsize(os.path.join(PACK, "graph.bin")),
                  "layout": "u32le head[T] | u32le tail[T] | u8 rel[T]"},
        "weights": {"path": "weights.bin", "bytes": int(w.nbytes), "dtype": "f32le", "layout": layout},
    },
    "linker": {"type": "longest-name-v1", "weak_relations": ["has_tags"],
               "note": "names from the pack's own entity list; an entity whose every edge is a weak relation is used only "
                       "when nothing else matches"},
    "scorer": {"type": "operator-path-v1", "encoder": "all-MiniLM-L6-v2", "tok_dim": 384, "max_tokens": 32,
               "mask_token": "NE", "dim": 16, "max_hops": 3, "top_k_walk": 20, "no_revisit": True,
               "tau": float(sd["logtau"].exp()), "tagger_classes": 2 * len(K.RELS) + 1,
               "source": "KG3b R2 (= KG10B C0 = KG3c ctl), seed 0, MetaQA 1+2-hop train",
               "params_trained": int(model.nparams())},
}
for k in ("entities", "graph", "weights"):
    manifest["files"][k]["sha256"] = sha(os.path.join(PACK, manifest["files"][k]["path"]))
json.dump(manifest, open(os.path.join(PACK, "manifest.json"), "w"), indent=1)

# ------------------------------------------------------------------ MiniLM ONNX (Xenova export), split into < 15 MB parts
# fp16 (default): token vectors match the fp32 PyTorch model the scorer was trained on (cos 1.0000).
# q8 ("small"): half the download, cos ~0.99 mean / 0.966 min; offered as an option.
MD = os.path.join(a.site, "models/glm")
os.makedirs(MD, exist_ok=True)
CH = 9 * 1024 * 1024  # 9 MB parts: < 15 MB binary, and < 16 MB as base64 text for hosts that only serve text
enc_man = {}
for name, src in (("fp16", a.onnx), ("q8", a.onnx_small)):
    raw = open(src, "rb").read()
    nparts = (len(raw) + CH - 1) // CH
    for i in range(nparts):
        open(os.path.join(MD, f"minilm_{name}.part{i}"), "wb").write(raw[i * CH:(i + 1) * CH])
    enc_man[name] = {"source": "Xenova/all-MiniLM-L6-v2 onnx/" + os.path.basename(src), "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest(), "parts": [f"minilm_{name}.part{i}" for i in range(nparts)]}
vocab = a.vocab or os.path.join(os.path.dirname(os.path.dirname(a.onnx)), "vocab.txt")
shutil.copy(vocab, os.path.join(MD, "vocab.txt"))
json.dump({"default": "fp16", "variants": enc_man, "vocab": "vocab.txt", "license": "Apache-2.0 (sentence-transformers/all-MiniLM-L6-v2)"},
          open(os.path.join(MD, "manifest.json"), "w"), indent=1)

# ------------------------------------------------------------------ Python reference answers
from transformers import AutoModel, AutoTokenizer  # noqa: E402
EX = json.load(open(os.path.join(a.site, "glm/examples.json")))["examples"]
qs = []
rng = random.Random(2026)
for hop, n in ((1, 7), (2, 7), (3, 6)):  # TRAIN split only: no dev / held / test question is read here
    lines = [l.rstrip("\n") for l in open(os.path.join(DATA, f"{hop}-hop/qa_train.txt"), encoding="utf-8") if l.strip()]
    for l in rng.sample(lines, n):
        q, ans = l.split("\t")
        qs.append({"src": f"train-{hop}hop", "q": q, "hop": hop})
for e in EX:
    qs.append({"src": "example", "q": e["bracketed"], "hop": e["hop"]})
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
Lmax = 32
masked = [K.mask_q(x["q"]) for x in qs]
with torch.no_grad():
    b = tok(masked, padding="max_length", truncation=True, max_length=Lmax + 2, return_tensors="pt")
    h = enc(**b).last_hidden_state[:, 1:Lmax + 1]
    M = b["attention_mask"][:, 1:Lmax + 1].bool() & (b["input_ids"][:, 1:Lmax + 1] != tok.sep_token_id)
    T = torch.tensor(h.numpy().astype(np.float16).astype(np.float32))  # training used fp16-stored vectors
    lg = model.logits(T, M)
envq = []
for x in qs:
    t = K.BR.search(x["q"]).group(1)
    envq.append({"tid": kb.ent[t], "aids": set()})
env = Env(kb, envq, True)
out = []
for i, x in enumerate(qs):
    tid = envq[i]["tid"]
    mk = torch.tensor(candidates(kb, tid))
    l = lg[i].masked_fill(~mk, -1e9)
    order = torch.argsort(l, descending=True)[:20].numpy()
    path, _ = answer(env, i, order)
    fr = env.frontier(i, tuple(path)).tocsr()
    mx = fr.data.max() if fr.nnz else 0
    ans = sorted(kb.names[j] for j, v in zip(fr.indices, fr.data) if v == mx and j != tid)
    top1 = kb.top1(tid, fr)
    out.append({"src": x["src"], "q": x["q"], "hop": x["hop"], "masked": masked[i], "topic": kb.names[tid],
                "path": [K.EDGES[e] for e in path], "answers_top": ans, "top1": kb.names[top1] if top1 >= 0 else None,
                "n_frontier": int(fr.nnz), "n_cands": int(mk.sum()),
                "top5": [[[K.EDGES[e] for e in PATHS[j]], round(float(l[j]), 4)] for j in order[:5]],
                "tokvec": h[i][M[i]].numpy().round(5).tolist() if i < 5 else None})
os.makedirs(os.path.join(a.site, "tools/glm_parity"), exist_ok=True)
json.dump(out, open(os.path.join(a.site, "tools/glm_parity/ref.json"), "w"))
print("pack", {k: manifest["files"][k]["bytes"] for k in ("entities", "graph", "weights")}, "entities", kb.n, "triples", len(tr))
for r in out:
    print(r["src"], "|", r["q"], "->", " > ".join(r["path"]), "->", r["answers_top"][:4])
