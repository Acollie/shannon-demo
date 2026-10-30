# Shannon knowledge pack format (`shannon-pack/1`)

A **pack** is one loadable graph plus everything Shannon needs to answer questions over it: the entity names (for the
linker), the relations, the triples, and the trained scorer weights. The GLM page (`glm.html`) loads and unloads packs at
run time on the main thread (`assets/glm/engine.js`, core in `assets/glm/glm.js`). Unloading a pack removes those facts
from Shannon; with no pack loaded it cannot answer anything. Several packs can be loaded at once: the linker runs over
every loaded pack and the question goes to the pack with the best entity match.

v0 ships one pack, `packs/movies` (MetaQA / WikiMovies, built by `tools/glm_build.py`). The next pack is a general
Wikidata pack built from the rung-10 model (KG15, Mintaka). This file is the contract for it.

## Directory

```
packs/<id>/
  manifest.json     required; describes everything below
  entities.txt      required; UTF-8, one canonical name per line, line i = entity id i (ids are 0..N-1)
  aliases.txt       optional; UTF-8, "alias<TAB>entity id" per line (Wikidata aliases, labels in other casing, ...)
  graph.bin         required; the triples (layout below)
  weights.bin       required; float32 little-endian scorer / tagger / linker weights (layout in the manifest)
  LICENSE.txt       required; the data licence text
  (more files)      allowed when a scorer or linker type needs them; list them under manifest.files
```

Pack ids are listed for the page in `glm/packs.json` (`status: "available" | "planned"`, `default: true` loads on start).

## manifest.json

```jsonc
{
  "format": "shannon-pack/1",
  "id": "movies",                                  // = directory name
  "title": "Movies (MetaQA / WikiMovies)",
  "description": "...",
  "license": "CC BY 3.0",
  "attribution": "...",                            // shown on the page
  "relations": [                                   // R relations; index = relation id used in graph.bin
    {"id": "directed_by", "label": "directed by", "inverse_label": "director of"}
  ],
  "inverse_suffix": "^-1",
  "counts": {"entities": 43234, "triples": 133582},
  "files": {
    "entities": {"path": "entities.txt", "bytes": 0, "sha256": "..."},
    "aliases":  {"path": "aliases.txt",  "bytes": 0, "sha256": "..."},          // optional
    "graph":    {"path": "graph.bin", "bytes": 0, "sha256": "...", "rel_dtype": "u8",   // "u8" (R <= 255) or "u16"
                 "layout": "u32le head[T] | u32le tail[T] | u8 rel[T]"},
    "weights":  {"path": "weights.bin", "bytes": 0, "sha256": "...", "dtype": "f32le",
                 "layout": {"<name>": {"offset": 0, "shape": [384]}}}           // offset in float32 elements
  },
  "linker": {"type": "longest-name-v1", "weak_relations": ["has_tags"]},
  "scorer": {"type": "operator-path-v1", ...}     // fields depend on the type, below
}
```

### Edge types

Relation `r` gives two edge types: `2r` (head to tail, named `id`) and `2r+1` (tail to head, named `id + inverse_suffix`).
The engine builds one CSR adjacency per edge type from `graph.bin` when the pack loads (`Pack` constructor in `glm.js`).

### graph.bin

`T` = `counts.triples`. Three arrays back to back, no header: `head` (uint32 LE, T), `tail` (uint32 LE, T),
`rel` (uint8 or uint16 LE, T). Duplicates should be removed. Triples sorted by (head, rel, tail) compress best.

## Linker types

- **`longest-name-v1`** (v0). Normalise names and the question the same way (Unicode NFD, drop combining marks,
  lowercase, keep runs of letters and digits, join with one space). Find the longest run of question words that is a
  name or alias. Ties: an entity with at least one non-weak edge beats an entity whose edges are all `weak_relations`,
  then higher degree. An explicit `[Name]` in the question wins. The matched span is replaced by `scorer.mask_token`
  before encoding.
- **`kg15-mlp-v1`** (planned, for the Wikidata pack). KG15's learned linker: the exact channel above (word n-grams 1..6
  against normalised labels + aliases, up to 50 entities per key by degree) plus, optionally, the dense channel; the
  14 features and the 14 -> 32 -> 1 MLP weights go in `weights.bin` (names `link_W1`, `link_b1`, `link_W2`, `link_b2`),
  feature order documented in `manifest.linker.features`. The dense channel needs MiniLM vectors of entity labels
  (1M x 384 float is ~1.5 GB, too big for a phone); ship it only as an int8 / PCA-reduced shard set, or leave it out and
  report the exact-channel-only recall on Mintaka dev.

## Scorer types

- **`operator-path-v1`** (v0, KG3b R2 = KG10B C0). Fields: `encoder` (`all-MiniLM-L6-v2`), `tok_dim` 384,
  `max_tokens` 32, `mask_token` "NE", `dim` D = 16, `max_hops` 3, `top_k_walk` 20, `no_revisit` true, `tau`,
  `tagger_classes` = 2R+1. Weights: `mu`, `sd` (384), `W` (2R+1 x 384), `b` (2R+1), `Le` (2R x D x D: log-operator per
  edge type, `Le[2r+1] = Le[2r]^T`), `Me` (2R x D x D: `expm(Le)`, precomputed in float64).
  Question: MiniLM token vectors of the masked question (no [CLS]/[SEP]), `softmax(W (x - mu)/sd + b)` per token, sum the
  first 2R classes over tokens = soft relation counts `p`; `Q = expm(sum_e p_e Le[e])`.
  Paths: every edge sequence of length 1..max_hops that is executable from the topic (frontier non-empty after each step
  with the topic removed; for the last step, some node of the previous frontier has that edge type). Path vector = product
  `Me[e1] Me[e2] ...`. Score = `tau * cos(Q, path)`. Walk the top `top_k_walk` paths in order; the first one with a
  non-empty frontier answers; answers ranked by number of realising paths.
- **`s1-sym-v1`** (planned, KG15 / KG9 "S1-sym"). The relation operators come from relation *names* read by the question
  reader; in a pack they are fixed, so precompute them offline and ship the operator bank (`Me`, 2R x D x D) plus the
  question reader's weights. Extra fields: `max_hops` 2, `cand_cap` 3000, `beta` (weight of the linker logit),
  `top_topics` 3, and the H1 answer-type head (`h1_W` 6 x 384, `h1_b` 6 on the pooled MiniLM sentence vector) with the
  six operations (set, count, yes/no, argmax, argmin, compare). Literal attributes for argmax / argmin / compare go in an
  optional `literals.bin` (entity id, attribute relation id, float32 value), listed under `files`. Adding this type
  means porting the reader and H1 to `glm.js` and checking parity against the Python pipeline on Mintaka train/dev
  questions (never test), like `tools/glm_parity/`.

## Size and sharding

v0 loads a whole pack into memory (movies: 1.9 MB). For a large graph (Wikidata5M: 4.6M entities, 21M triples) keep the
same manifest and add `"shards"`: a list of `{id, entities: [lo, hi), path, bytes}` where each shard holds the
adjacency of its entities (same `graph.bin` layout, entity ids global) and a name-index shard by normalised-name prefix.
Target 1-2 MB per shard, fetched on demand and kept in Cache Storage / IndexedDB; unloading a shard is "unloading an
idea". The engine interface stays the same: `load(id)`, `unload(id)`, `ask(question)`.

## Checks a new pack must pass before it is listed as available

1. `sha256` of every file matches the manifest.
2. Parity: the browser pipeline (Node, `tools/glm_parity/parity.mjs` adapted to the pack) gives the same linked topic,
   path and answers as the Python pipeline on at least 20 train or dev questions (never test or held questions).
3. Token-vector check: the shipped encoder's token vectors against the vectors the scorer was trained on, cosine > 0.99.
4. Licence and attribution present (Wikidata: CC0; Wikidata5M: check before shipping).
