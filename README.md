# Shannon demo

Live: https://acollie.github.io/shannon-demo/

Shannon is a small research language model that must pass everything through **32 discrete idea codes** (a 4,096-entry codebook):
text → frozen MiniLM encoder → 32 codes → idea transformer → decoder → next 128 word-pieces. The decoder never sees the text.
This site runs it entirely in the browser with [onnxruntime-web](https://onnxruntime.ai/) 1.20.1 (WASM, from jsDelivr), and shows:

- text → idea codes → text, with sampling settings, per-slot edits (swap a code for a neighbour) and ablations (all-modal, shuffled order)
- the idea map (UMAP of the codebook, k-means regions, the "thought path" of your input)
- the idea dictionary (top spans, distinctive words, contexts, slot usage, nearest codes)
- results vs a plain encoder-decoder baseline, reported honestly: **Shannon is worse** (10M: 1.478 vs 1.407 bits per byte)

All text on the site is public FineWeb-Edu.

## Layout

```
index.html, assets/          static front end (no build step)
models/models.json           {"current": "<model-id>", "models": [...]}; ?model=<id> picks another
models/<id>/                 encoder.onnx ideas.onnx decoder.onnx export.json   (tools/export_model.py)
                             dictionary.json map.json presets.json stats.json  (tools/build_data.py, then tools/relabel_regions.py)
                             vocab.txt (BERT-uncased, from ~/shannon/assets/vocab.txt), card.json (hand-written, see below)
tools/                       export / data scripts (run on the GPU PC, CPU only)
models/src/                  local checkpoint copies, git-ignored
```

## Swapping in a new model (e.g. 20M)

Run everything on the GPU PC (`ssh gpu`), CPU only and `nice -n 19`; never touch the training runs or chain scripts.
The PC has a separate venv for this, `~/demo-venv` (CPU torch, onnx, onnxruntime, umap-learn, scikit-learn).

1. Copy the tools: `scp tools/*.py gpu:~/shannon-demo-tools/`
2. Make held-out windows from the shard's validation split (the same split training evaluates on):
   `cd ~/shannon && nice -n 19 ~/demo-venv/bin/python ~/shannon-demo-tools/make_windows.py --shannon-src ~/shannon --shard data/shards/fw600m --out ~/shannon-demo-out/heldout_fw600m.npz`
3. Export (writes the three .onnx files + export.json; picks int8 for ideas/decoder if BPB moves < 1%):
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
