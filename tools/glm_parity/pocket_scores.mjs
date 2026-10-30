// GLM Pocket calibration data: the browser engine (assets/glm/wiki.js, the same code the page runs) over the Mintaka DEV
// questions in tools/glm_parity/ref_general_2000.json, recording per question the path score the page sees (top path
// score), the margin to the second path, the answer type, and whether the answer was right. "Right" = the browser answer
// is identical to the Python reference answer AND the reference's gold check says hit (same rule as accuracy_general.mjs).
// Inference only; dev questions only (no sealed data).
//   cd <dir with onnxruntime-web@1.20.1> && node <site>/tools/glm_parity/pocket_scores.mjs <site> [out.json]
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";

const SITE = path.resolve(process.argv[2] || new URL("../..", import.meta.url).pathname);
const OUT = process.argv[3] || path.join(SITE, "tools/glm_parity/pocket_scores_dev.json");
const { Encoder } = await import(SITE + "/assets/glm/glm.js");
const { WikiPack } = await import(SITE + "/assets/glm/wiki.js");
ort.env.wasm.numThreads = 1;
const rd = f => fs.readFileSync(path.join(SITE, f));
const mm = JSON.parse(rd("models/glm/manifest.json"));
const enc = await new Encoder(ort, rd("models/glm/vocab.txt").toString()).init(Buffer.concat(mm.variants[mm.default].parts.map(p => rd("models/glm/" + p))));
const pack = await new WikiPack(JSON.parse(rd("packs/general/manifest.json")), async p => new Uint8Array(rd("packs/general/" + p))).init();
const ref = JSON.parse(rd("tools/glm_parity/ref_general_2000.json"));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const pyAns = r => (r.node !== null && r.node !== undefined ? { node: r.node } : r.answer);
const jsAns = o => (o.node !== null && o.node !== undefined && o.node >= 0 ? { node: o.node } : (o.answer && o.answer.node !== undefined ? { node: o.answer.node } : o.answer));
const rows = [];
const T0 = performance.now();
for (const r of ref.rows) {
  const o = await pack.ask(r.q, enc, {});
  const s = o.top5 || [];
  rows.push({ id: r.id, type: r.type, op: o.op, ok: !!o.ok && (o.answers || []).length > 0, score: o.score,
    margin: s.length > 1 ? s[0][1] - s[1][1] : null, same: same(jsAns(o), pyAns(r)), hit: same(jsAns(o), pyAns(r)) ? r.hit : 0 });
  if (rows.length % 200 === 0) console.log(rows.length, ((performance.now() - T0) / 1000).toFixed(0) + " s");
}
fs.writeFileSync(OUT, JSON.stringify({ note: "Per-question path score + correctness of the browser engine on Mintaka dev (2,000). Built by tools/glm_parity/pocket_scores.mjs.", n: rows.length, rows }));
console.log("wrote", OUT);
