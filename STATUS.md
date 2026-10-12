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

## In progress
- build_data.py for 20M on the PC (~/shannon-demo-out/build20m.log), then scp to models/20m-fineweb, relabel_regions, vocab.txt, set models.json current=20m-fineweb, test, push, verify live.

## Notes
- PC venv for this work: ~/demo-venv (separate from ~/shannon/.venv). Never touch runs/poc20m_*, ts20m_*, chain*.sh, data/shards.
