// GLM v1 parity, general pack: the browser pipeline (assets/glm/wiki.js + glm.js Encoder + onnxruntime-web WASM + the
// shipped MiniLM) in Node against the Python reference (tools/glm_general/ref.py -> tools/glm_parity/ref_general.json:
// the KG16 R0 pipeline in PyTorch on the same pack). Mintaka DEV questions only.
//   cd <dir with onnxruntime-web@1.20.1 installed> && node <site>/tools/glm_parity/parity_general.mjs <site> [fp16|q8] [n]
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";

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
const pack = await new WikiPack(man, async p => new Uint8Array(rd("packs/general/" + p))).init();
const ref = JSON.parse(rd("tools/glm_parity/ref_general.json"));
const N = Math.min(+(process.argv[4] || ref.rows.length), ref.rows.length);
console.log("encoder", VAR, "pack init", pack.buildMs.toFixed(0), "ms; questions", N);

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const pyAns = r => (r.node !== null && r.node !== undefined ? { node: r.node } : r.answer);
const jsAns = o => (o.node !== null && o.node !== undefined && o.node >= 0 ? { node: o.node } : (o.answer && o.answer.node !== undefined ? { node: o.answer.node } : o.answer));
let nT = 0, nOp = 0, nP = 0, nA = 0, nBoth = 0, hitsPy = 0, hitsJs = 0;
const rows = [], times = [];
for (const r of ref.rows.slice(0, N)) {
  const o = await pack.ask(r.q, enc, { core: true });
  times.push(o.timings.total_ms);
  const okT = same(o.topics || [], r.topic_ids), okO = (o.op || null) === r.op;
  const pyTop = r.top.length ? r.top[0] : null, okP = same(o.top, pyTop);
  const okA = same(jsAns(o), pyAns(r));
  nT += okT; nOp += okO; nP += okP; nA += okA; nBoth += okP && okA;
  hitsPy += r.hit; hitsJs += okA ? r.hit : 0;
  const row = { id: r.id, q: r.q, topics_ok: okT, op_ok: okO, path_ok: okP, answer_ok: okA, ms: +o.timings.total_ms.toFixed(1) };
  if (!(okT && okP && okA)) {
    Object.assign(row, { py: { topics: r.topic_ids, tlog: r.tlog, op: r.op, top: r.top, ans: pyAns(r) }, js: { topics: o.topics, tlog: o.tlog, op: o.op, top: o.top3, ans: jsAns(o) } });
    console.log("DIFF", r.id, r.q, JSON.stringify(row.py), "\n    js", JSON.stringify(row.js));
  }
  rows.push(row);
}
times.sort((a, b) => a - b);
const pct = x => +(100 * x / N).toFixed(1);
const summary = {
  encoder: VAR, n: N, split: "Mintaka dev (first N)",
  same_topics: nT, same_op: nOp, same_top_path: nP, same_answer: nA, same_path_and_answer: nBoth,
  pct_top_path: pct(nP), pct_answer: pct(nA), pct_path_and_answer: pct(nBoth),
  py_hits1: pct(hitsPy), node_ms_median: +times[Math.floor(N / 2)].toFixed(1), node_ms_max: +times[N - 1].toFixed(1),
  wall_s: +((performance.now() - T0) / 1000).toFixed(1),
};
console.log(JSON.stringify(summary, null, 1));
fs.writeFileSync(path.join(SITE, `tools/glm_parity/parity_general_${VAR}.json`), JSON.stringify({ summary, rows }, null, 1));
