// GLM v1 parity, general pack: the browser pipeline (assets/glm/wiki.js + glm.js Encoder + onnxruntime-web WASM + the
// shipped MiniLM) in Node against the Python reference (tools/glm_general/ref.py -> tools/glm_parity/ref_general_2000.json:
// the KG16 R0 pipeline in PyTorch on the same pack). Mintaka DEV questions only.
//   cd <dir with onnxruntime-web@1.20.1 installed> && node <site>/tools/glm_parity/accuracy_general.mjs <site> [fp16|q8] [n] [tag]
// Also records per-question time (median, p90) and bytes: "session" = pack bytes fetched by that question with the
// files earlier questions fetched still cached (one visitor asking all 2,000 in a row); "cold" = all pack files that
// question touches, as if it were the first question of a visit (after the one-off start-up files). Bytes are raw pack
// bytes and gzip -6 sizes (GitHub Pages serves the shards gzip-compressed).
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";
import zlib from "zlib";

const SITE = path.resolve(process.argv[2] || new URL("../..", import.meta.url).pathname);
const { Encoder } = await import(SITE + "/assets/glm/glm.js");
const { WikiPack } = await import(SITE + "/assets/glm/wiki.js");
ort.env.wasm.numThreads = 1;
const rd = f => fs.readFileSync(path.join(SITE, f));
const mm = JSON.parse(rd("models/glm/manifest.json"));
const VAR = process.argv[3] || mm.default;
const T0 = performance.now();
const enc = await new Encoder(ort, rd("models/glm/vocab.txt").toString()).init(Buffer.concat(mm.variants[VAR].parts.map(p => rd("models/glm/" + p))));
const man = JSON.parse(rd("packs/general/manifest.json"));
const gz = new Map(), gzOf = p => { if (!gz.has(p)) gz.set(p, zlib.gzipSync(rd("packs/general/" + p), { level: 6 }).length); return gz.get(p); };
let touched = null;
const pack = await new WikiPack(man, async p => new Uint8Array(rd("packs/general/" + p))).init();
const startFiles = new Set(pack.files.keys());
{ // record every pack file a question needs, cached or not (the need* calls skip cached shards before fetching)
  const F = man.files, pad = (x, w) => String(x).padStart(w, "0");
  const up = (lo, v) => { let a = 0, b = lo.length; while (a < b) { const m = (a + b) >> 1; if (lo[m] <= v) a = m + 1; else b = m; } return a - 1; };
  const sh0 = pack.shards.bind(pack), nn0 = pack.needNames.bind(pack), ni0 = pack.needIdx.bind(pack), nl0 = pack.needLitVals.bind(pack);
  pack.shards = (kind, ids, cache) => { if (touched) for (const v of ids) touched.add(`${F[kind].dir}/${pad(up(F[kind].lo, v), 3)}.bin`); return sh0(kind, ids, cache); };
  pack.needNames = ids => { if (touched) for (const v of ids) touched.add(`${F.names.dir}/${pad(Math.floor(v / F.names.per_shard), 3)}.txt`); return nn0(ids); };
  pack.needIdx = keys => { if (touched) for (const k of keys) touched.add(`${F.idx.dir}/${pad(pack.idxShard(k), 4)}.txt`); return ni0(keys); };
  pack.needLitVals = () => { if (touched) touched.add(F.litval.path); return nl0(); };
}
const REF = process.env.GLM_REF || "tools/glm_parity/ref_general_2000.json";
const ref = JSON.parse(fs.readFileSync(path.isAbsolute(REF) ? REF : path.join(SITE, REF)));
const N = Math.min(+(process.argv[4] || ref.rows.length), ref.rows.length);
console.log("encoder", VAR, "pack init", pack.buildMs.toFixed(0), "ms; questions", N);

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const pyAns = r => (r.node !== null && r.node !== undefined ? { node: r.node } : r.answer);
const jsAns = o => (o.node !== null && o.node !== undefined && o.node >= 0 ? { node: o.node } : (o.answer && o.answer.node !== undefined ? { node: o.answer.node } : o.answer));
let nT = 0, nOp = 0, nP = 0, nA = 0, nBoth = 0, hitsPy = 0, hitsJs = 0;
const rows = [], times = [], bS = [], bSg = [], bC = [], bCg = [];
const sz = p => rd("packs/general/" + p).length;
for (const r of ref.rows.slice(0, N)) {
  touched = new Set();
  const before = new Set(pack.files.keys());
  const o = await pack.ask(r.q, enc, { core: true });
  times.push(o.timings.total_ms);
  const cold = [...touched].filter(p => !startFiles.has(p)), fresh = [...pack.files.keys()].filter(p => !before.has(p));
  bS.push(fresh.reduce((a, p) => a + sz(p), 0)); bSg.push(fresh.reduce((a, p) => a + gzOf(p), 0));
  bC.push(cold.reduce((a, p) => a + sz(p), 0)); bCg.push(cold.reduce((a, p) => a + gzOf(p), 0));
  const okT = same(o.topics || [], r.topic_ids), okO = (o.op || null) === r.op;
  const pyTop = r.top.length ? r.top[0] : null, okP = same(o.top, pyTop);
  const okA = same(jsAns(o), pyAns(r));
  nT += okT; nOp += okO; nP += okP; nA += okA; nBoth += okP && okA;
  hitsPy += r.hit; hitsJs += okA ? r.hit : 0;
  const row = { id: r.id, q: r.q, topics_ok: okT, op_ok: okO, path_ok: okP, answer_ok: okA, ms: +o.timings.total_ms.toFixed(1), cold_bytes: bC[bC.length - 1], cold_gz: bCg[bCg.length - 1] };
  if (!(okT && okP && okA)) {
    Object.assign(row, { py: { topics: r.topic_ids, tlog: r.tlog, op: r.op, top: r.top, ans: pyAns(r) }, js: { topics: o.topics, tlog: o.tlog, op: o.op, top: o.top3, ans: jsAns(o) } });
    console.log("DIFF", r.id, r.q, JSON.stringify(row.py), "\n    js", JSON.stringify(row.js));
  }
  rows.push(row);
}
const tsum = times.reduce((a, b) => a + b, 0);
times.sort((a, b) => a - b);
const med = a => { const b = a.slice().sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
const mean = a => Math.round(a.reduce((x, y) => x + y, 0) / a.length);
const pct = x => +(100 * x / N).toFixed(1);
const summary = {
  browser_hits1: pct(hitsJs), // hit = browser answer identical to the Python answer, whose gold hit is recorded in the ref file
  note: "Browser engine (Node + onnxruntime-web WASM + shipped MiniLM) over ALL Mintaka dev questions with the 303k general pack. hits are counted from the Python reference gold check when the browser answer is identical; any non-identical answer is counted as a miss.",
  encoder: VAR, n: N, split: "Mintaka dev (first N)",
  same_topics: nT, same_op: nOp, same_top_path: nP, same_answer: nA, same_path_and_answer: nBoth,
  pct_top_path: pct(nP), pct_answer: pct(nA), pct_path_and_answer: pct(nBoth),
  py_hits1: pct(hitsPy), node_ms_median: +times[Math.floor(N / 2)].toFixed(1), node_ms_p90: +times[Math.floor(0.9 * N)].toFixed(1),
  node_ms_mean: +(tsum / N).toFixed(1), node_ms_max: +times[N - 1].toFixed(1),
  bytes: { session_mean_raw: mean(bS), session_mean_gzip: mean(bSg), cold_median_raw: med(bC), cold_median_gzip: med(bCg), cold_mean_raw: mean(bC), cold_mean_gzip: mean(bCg) },
  machine: process.env.GLM_MACHINE || null, ref_file: REF,
  wall_s: +((performance.now() - T0) / 1000).toFixed(1),
};
console.log(JSON.stringify(summary, null, 1));
const TAG = process.argv[5] || VAR;
fs.writeFileSync(process.env.GLM_OUT || path.join(SITE, `tools/glm_parity/accuracy_general_2000_${TAG}.json`), JSON.stringify({ summary, rows }, null, 1));
