# STATUS (resume point)

Updated: 2026-10-12

## Done
- tools/export_model.py, tools/build_data.py (from the earlier agent), tools/relabel_regions.py (new: fixes region labels / stop-words in map.json + dictionary.json without the model).
- models/10m-fineweb: ONNX (encoder fp16w 46 MB, ideas int8 17 MB, decoder int8 30 MB) + dictionary/map/presets/stats + vocab.txt + card.json (BPB vs plain baseline from the training logs on the PC).
- Old draft data/10m removed (superseded by models/10m-fineweb).

- Front end (index.html, assets/app.js, tokenizer.js, style.css); tested locally in headless Chromium: model loads, encode 85 ms, ~5 ms/token, no console errors, no horizontal scroll at 375px.
- tools/make_windows.py (held-out val windows for re-exports). PC venv ~/demo-venv (CPU torch, onnx, ort, umap) being installed.

- LIVE: https://acollie.github.io/shannon-demo/ (repo Acollie/shannon-demo, Pages from main /). Verified live in headless Chromium with the 10M model: loads 94 MB, encodes, generates ~5 ms/token, no console errors.
- JS tokenizer verified: re-tokenising 293 held-out windows reproduces the training token ids exactly in 278; the 15 misses all start mid-word (## piece), which is expected.
- 20M export (PC, ~/shannon-demo-out/20m-fineweb): every int8 decoder variant moved BPB 2.5-3.4% (tools/quant_sweep.py), so 20M ships "mixed": encoder fp16w 46.5 MB, ideas int8 per-channel 31 MB, decoder fp16w 91.5 MB (BPB drift 0.02%). Held-out windows: ~/shannon-demo-out/heldout_fw600m.npz (fw600m val split, 2049 windows; same windows the 10M data used).
- models/20m-fineweb/card.json written from poc20m_{shannon,t5}/metrics.jsonl (1.3198 vs 1.2704).

- 20M swapped in: models/20m-fineweb (dictionary/map/presets/stats from build_data on the PC; held-out BPB fp32 1.305 / shuffle 1.318 / null 1.354; all 4096 codes used). models.json current=20m-fineweb; 10M kept at ?model=10m-fineweb. Tested locally (loads 169 MB, ~5 ms/token, no console errors).

- 20M verified LIVE (commit a2aa584): loads 169 MB, 32 codes in ~80 ms, ~5 ms/token, no console errors. PC training (ts20m_t5) kept progressing throughout; no demo jobs left running on the PC.

- 2026-10-13: **Shannon QA is now the main page** (index.html); the idea demo moved to ideas.html (links both ways). JS port of treaty-graph SystemX,
  models in models/qa (214 MB total, fp16w + int8 word table for yes/no). Parity vs Python SystemX 66/66. ~0.8 s/question in WASM on the Mac.
  PC scratch for this: ~/qa-demo (src models, out/ ONNX variants, demo-venv got transformers 5.19.0). Nothing left running on the PC.

## Open / optional
- Shannon QA download is 214 MB (yesno 95 MB is the big one). A reader int8 file would save 33 MB at the cost of 1/66 parity.
- The 20M download is heavy for phones (169 MB; decoder 91 MB because every int8 decoder variant drifted 2.5-3.4% BPB). A future option: find the drifting decoder ops (static int8 with calibration, or MatMulNBits weight-only int8) to get it near 60 MB.
- GitHub warns that decoder.onnx (87 MiB) is above the 50 MB recommendation; it is under the 100 MB hard limit.

## Notes
- PC venv for this work: ~/demo-venv (separate from ~/shannon/.venv). Never touch runs/poc20m_*, ts20m_*, chain*.sh, data/shards.
