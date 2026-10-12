# STATUS (resume point)

Updated: 2026-10-12

## Done
- tools/export_model.py, tools/build_data.py (from the earlier agent), tools/relabel_regions.py (new: fixes region labels / stop-words in map.json + dictionary.json without the model).
- models/10m-fineweb: ONNX (encoder fp16w 46 MB, ideas int8 17 MB, decoder int8 30 MB) + dictionary/map/presets/stats + vocab.txt + card.json (BPB vs plain baseline from the training logs on the PC).
- Old draft data/10m removed (superseded by models/10m-fineweb).

- Front end (index.html, assets/app.js, tokenizer.js, style.css); tested locally in headless Chromium: model loads, encode 85 ms, ~5 ms/token, no console errors, no horizontal scroll at 375px.
- tools/make_windows.py (held-out val windows for re-exports). PC venv ~/demo-venv (CPU torch, onnx, ort, umap) being installed.

## In progress
- Create repo Acollie/shannon-demo, push, enable Pages, verify live.
- Then: 20M swap if ~/shannon/runs/poc20m_shannon/cooldown_1.0.pt exists on the PC (see README "Swapping in a new model").

## Notes
- Held-out windows npz used by the earlier agent for export/build_data was not found on disk; for a re-export, build one on the PC from the shard's val split (see README).
