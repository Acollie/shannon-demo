# Shannon demo

Live: https://acollie.github.io/shannon-demo/

Two in-browser demos, both run with [onnxruntime-web](https://onnxruntime.ai/) 1.20.1 (WASM, single thread, from jsDelivr); nothing is sent to a server.

- **`index.html`: Shannon QA.** Paste any text and ask questions about it. A JavaScript port of treaty-graph's no-hand-code QA system
  (`treaty-graph/src/treatygraph/generic.py`, `SystemX`): passages -> IDF overlap + MiniLM retrieval -> minilm-squad2 spans in
  generic forms (span / list / number / yes-no / none) -> DeBERTa-v3 yes/no cross-encoder -> question-form model + log-linear gate,
  which can refuse ("not answerable"). It never generates text. Shows the answer, its form, the gate margin, the cited passage with the span
  marked, and an answer-path graph (question -> passages -> candidates with gate scores -> choice).
- **`ideas.html`: Shannon's idea model** (described below).

## GLM v0 (`glm.html`): ask the movie graph

Shannon plus a knowledge graph, all in the browser (also on a phone): type a movie question, Shannon links the entity
(longest name match over the pack's own entity list), reads the question with frozen MiniLM token vectors, scores every
1-3 step relation path from that entity with 16x16 relation operators (KG3b R2 = KG10B C0, seed 0, 9,620 trained params),
walks the best one and shows the path, the next-best paths and per-step timing. Packs load and unload at run time
(`PACK_FORMAT.md`); v0 ships `packs/movies` (MetaQA / WikiMovies, CC BY 3.0, 1.9 MB).

```
glm.html, assets/glm/       app.js (UI), engine.js (main thread: ORT + packs), glm.js (pack, linker, scorer; also runs in Node)
packs/movies/               manifest.json, entities.txt, graph.bin, weights.bin, LICENSE.txt
models/glm/                 MiniLM-L6 ONNX (Xenova export) in 9 MB parts: fp16 (default, 45 MB) and q8 (23 MB), vocab.txt
vendor/ort/                 onnxruntime-web 1.20.1 WASM binary (the .mjs glue loads from jsDelivr)
glm/                        packs.json (catalog), examples.json (demo-written), sample.json (precomputed example)
tools/glm_build.py          builds the pack + encoder parts + Python reference (tools/glm_parity/ref.json)
tools/glm_parity/           parity.mjs (Node, onnxruntime-web) and results
```

Parity (JS vs the Python KG3b pipeline, 20 MetaQA train questions + 6 examples, JS given the question without brackets):
26/26 same linked entity, path and answers, with both encoders, in Node and in headless Chromium. Token vectors vs the fp32
PyTorch MiniLM the scorer was trained on: fp16 cos 1.0000; q8 mean 0.990, min 0.966.

## Shannon QA

```
index.html, assets/qa/          app.js (UI), shannon_qa.js (the SystemX port), engine.js (ORT), tokenizers.js
                                (BERT WordPiece with offsets + DeBERTa Unigram), npsort.js (numpy argsort port), qa.css
models/qa/                      embed.onnx 45.9 MB, reader.onnx 68.1 MB, yesno.onnx 95.1 MB, yesno_vocab.json, vocab.txt,
                                gate.json (= treaty-graph data/hybrid/gate_x.json), manifest.json (sizes, sha256, precision)
qa/docs/, qa/examples.json      example texts + example questions (written for the demo, no exam questions)
qa/parity_extra.json            30 more demo-written questions, used only for parity testing
tools/qa_export.py              ONNX export (torch.export / dynamo) + quantisation variants + checks (run on the GPU PC, CPU)
tools/qa_parity/                ref_dump.py (runs the Python SystemX, dumps every intermediate), parity.mjs (runs the JS port in
                                Node with onnxruntime-web), test_tok.mjs + tok_fix.py (tokeniser fixtures), reference dumps
```

Total download ~214 MB, once; files are kept in Cache Storage (`shannon-qa-v1`, keyed by sha256) and the page auto-loads them on later visits.

**Parity (JS vs Python `SystemX`):** 66/66 identical answers and forms (36 example questions + 30 extra, over the three example texts),
with the shipped files. Max drift vs PyTorch fp32: gate margin 0.05, P(yes) 0.014, reader margin 0.008. Tokenisers reproduce HF token ids and
offsets exactly on 549 strings incl. odd Unicode; numpy's argsort tie order (introsort, small-sort threshold 15 in numpy 2.5) is ported exactly,
which matters because overlap scores tie often.

**Precision:** embed and reader store fp16 weights (fp32 compute); yesno stores its 128k-word embedding table as int8 and the rest as fp16.
Full dynamic int8 was tried: in WASM it changed 1/66 answers (reader int8) and moved P(yes) by up to 0.49 (yes/no int8), so it is not used.
(On the GPU PC, ORT's CPU int8 numbers are much worse because the Ryzen 3700X has no VNNI; judge int8 in WASM, not there.)

Re-running parity:
```
cd treaty-graph && uv run python ../../shannon-demo/tools/qa_parity/ref_dump.py <examples.json> <docs dir> ref.json   # Python reference
cd <dir with onnxruntime-web@1.20.1 installed> && node tools/qa_parity/parity.mjs ref.json shipped
```

## Shannon idea model (`ideas.html`)

Shannon is a small research language model that must pass everything through **32 discrete idea codes** (a 4,096-entry codebook):
text → frozen MiniLM encoder → 32 codes → idea transformer → decoder → next 128 word-pieces. The decoder never sees the text.
This site runs it entirely in the browser with [onnxruntime-web](https://onnxruntime.ai/) 1.20.1 (WASM, from jsDelivr), and shows:

- text → idea codes → text, with sampling settings, per-slot edits (swap a code for a neighbour) and ablations (all-modal, shuffled order)
- the idea map (UMAP of the codebook, k-means regions, the "thought path" of your input)
- the idea dictionary (top spans, distinctive words, contexts, slot usage, nearest codes)
- results vs a plain encoder-decoder baseline, reported honestly: **Shannon is worse** at both sizes
  (10M: 1.478 vs 1.407 bits per byte; 20M: 1.320 vs 1.270)

Default model: `20m-fineweb` (≈169 MB download). The lighter 10M model (≈94 MB) is at `?model=10m-fineweb`.

All text on the idea page is public FineWeb-Edu.

## Layout

```
ideas.html, assets/          static front end (no build step)
models/models.json           {"current": "<model-id>", "models": [...]}; ?model=<id> picks another
models/<id>/                 encoder.onnx ideas.onnx decoder.onnx export.json   (tools/export_model.py)
                             dictionary.json map.json presets.json stats.json  (tools/build_data.py, then tools/relabel_regions.py)
                             vocab.txt (BERT-uncased, from ~/shannon/assets/vocab.txt), card.json (hand-written, see below)
tools/                       export / data scripts (run on the GPU PC, CPU only)
models/src/                  local checkpoint copies, git-ignored
```

## Swapping in a new model (how 20M was added)

Run everything on the GPU PC (`ssh gpu`), CPU only and `nice -n 19`; never touch the training runs or chain scripts.
The PC has a separate venv for this, `~/demo-venv` (CPU torch, onnx, onnxruntime, umap-learn, scikit-learn).

1. Copy the tools: `scp tools/*.py gpu:~/shannon-demo-tools/`
2. Make held-out windows from the shard's validation split (the same split training evaluates on):
   `cd ~/shannon && nice -n 19 ~/demo-venv/bin/python ~/shannon-demo-tools/make_windows.py --shannon-src ~/shannon --shard data/shards/fw600m --out ~/shannon-demo-out/heldout_fw600m.npz`
3. Export (writes the three .onnx files + export.json; picks int8 for ideas/decoder if BPB moves < 1%, else stops with fp32 files that may be too big;
   then run `tools/quant_sweep.py ... --variants per_channel,fp16w,mixed --pick <best>` on the `_fp32` it leaves behind. 20M uses `mixed`):
   `nice -n 19 ~/demo-venv/bin/python ~/shannon-demo-tools/export_model.py --shannon-src ~/shannon --ckpt ~/shannon/runs/poc20m_shannon/cooldown_1.0.pt --windows <npz> --out ~/shannon-demo-out/20m-fineweb --n-check 24`
4. Site data: `nice -n 19 ~/demo-venv/bin/python ~/shannon-demo-tools/build_data.py --shannon-src ~/shannon --ckpt <same> --windows <npz> --vocab ~/shannon/assets/vocab.txt --out ~/shannon-demo-out/20m-fineweb`
5. `scp -r gpu:~/shannon-demo-out/20m-fineweb models/`, then `python3 tools/relabel_regions.py models/20m-fineweb`,
   copy `vocab.txt`, and write `card.json` from the `"type": "cooldown", "budget": 1.0` lines of
   `runs/poc20m_shannon/metrics.jsonl` and `runs/poc20m_t5/metrics.jsonl` (copy the 10M card's fields).
6. Set `"current"` in `models/models.json` to the new id (keep the old id in `"models"` so `?model=10m-fineweb` still works).
7. Check each file is < 100 MB, serve locally (`python3 -m http.server`), test, commit, push; Pages redeploys in a minute or two.

## Notes

- Input is lower-cased BERT WordPiece; the JS tokenizer is `assets/tokenizer.js`. The last 128 word-pieces are used (a multiple of 4).
- Generation masks token ids below 999 (special/unused), with temperature, top-k, repetition penalty and a seed.
- GitHub Pages is not cross-origin isolated, so ONNX runs single-threaded WASM.
