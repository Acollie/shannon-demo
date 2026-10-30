// GLM parity: run the browser pipeline (assets/glm/glm.js + onnxruntime-web WASM + the shipped quantised MiniLM) in Node
// and compare with the Python pipeline's answers in ref.json (written by tools/glm_build.py).
//   cd <dir with onnxruntime-web@1.20.1 installed> && node <site>/tools/glm_parity/parity.mjs <site> [fp16|q8]
// The JS side gets the question WITHOUT the [brackets], so its own linker has to find the topic.
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";

const SITE = path.resolve(process.argv[2] || new URL("../..", import.meta.url).pathname);
const { Pack, Encoder, GLM } = await import(SITE + "/assets/glm/glm.js");
ort.env.wasm.numThreads = 1;
const rd = f => fs.readFileSync(path.join(SITE, f));
const mm = JSON.parse(rd("models/glm/manifest.json"));
const VAR = process.argv[3] || mm.default;
const enc = await new Encoder(ort, rd("models/glm/vocab.txt").toString()).init(Buffer.concat(mm.variants[VAR].parts.map(p => rd("models/glm/" + p))));
console.log("encoder", VAR);
const man = JSON.parse(rd("packs/movies/manifest.json"));
const ab = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
let t0 = performance.now();
const pack = new Pack(man, { entities: rd("packs/movies/entities.txt").toString(), graph: ab(rd("packs/movies/graph.bin")), weights: ab(rd("packs/movies/weights.bin")) });
console.log(`pack build ${(performance.now() - t0).toFixed(0)} ms, ${pack.n} entities`);
const glm = new GLM(enc); glm.load(pack);
const ref = JSON.parse(rd("tools/glm_parity/ref.json"));

// 1) token vectors: quantised ONNX (JS) vs fp32 PyTorch (Python), first 5 questions
const cosv = (a, b) => { let s = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; x += a[i] ** 2; y += b[i] ** 2; } return s / Math.sqrt(x * y); };
let cmin = 1, csum = 0, cn = 0;
for (const r of ref.filter(r => r.tokvec)) {
  const { tok, n, d } = await enc.tokens(r.masked);
  if (n !== r.tokvec.length) console.log("TOKEN COUNT DIFF", r.masked, n, r.tokvec.length);
  for (let t = 0; t < Math.min(n, r.tokvec.length); t++) { const c = cosv(tok.subarray(t * d, (t + 1) * d), r.tokvec[t]); cmin = Math.min(cmin, c); csum += c; cn++; }
}
console.log(`token vectors (${VAR} ONNX vs fp32 torch): mean cos ${(csum / cn).toFixed(4)}, min ${cmin.toFixed(4)} over ${cn} tokens`);

// 2) end to end
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const rows = []; let agreeAns = 0, agreePath = 0, agreeLink = 0, agreeMask = 0;
const times = [];
for (const r of ref) {
  const q = r.q.replace(/[\[\]]/g, "");
  const o = await glm.ask(q);
  times.push(o.timings.total_ms);
  const js = o.ok ? [...o.answers].sort() : null;
  const okL = o.link && o.link.name === r.topic, okM = o.masked === r.masked, okP = o.ok && same(o.path, r.path), okA = o.ok && same(js, r.answers_top);
  agreeLink += okL; agreeMask += okM; agreePath += okP; agreeAns += okA;
  rows.push({ src: r.src, q, link_ok: okL, path_ok: okP, answers_ok: okA, py: [r.path, r.answers_top.slice(0, 5)], js: o.ok ? [o.path, js.slice(0, 5)] : o.reason, ms: +o.timings.total_ms.toFixed(1) });
  if (!(okL && okP && okA)) console.log("DIFF", r.src, q, "\n  py", r.topic, "|", r.masked, "|", r.path.join(">"), r.answers_top.slice(0, 4), "\n  js", o.link && o.link.name, "|", o.masked, "|", o.ok && o.path.join(">"), js && js.slice(0, 4), "\n  py top5", JSON.stringify(r.top5), "\n  js top5", JSON.stringify(o.top5));
}
const N = ref.length, tr = rows.filter(r => r.src !== "example");
times.sort((a, b) => a - b);
const summary = {
  encoder: VAR, n: N, train_questions: tr.length, examples: N - tr.length,
  linker_same_topic: agreeLink, masked_text_same: agreeMask, same_path: agreePath, same_answers: agreeAns,
  train_same_answers: tr.filter(r => r.answers_ok).length, examples_same_answers: rows.filter(r => r.src === "example" && r.answers_ok).length,
  tokvec_cos_mean: +(csum / cn).toFixed(4), tokvec_cos_min: +cmin.toFixed(4),
  node_ms_median: +times[Math.floor(N / 2)].toFixed(1), node_ms_max: +times[N - 1].toFixed(1),
};
console.log(JSON.stringify(summary, null, 1));
fs.writeFileSync(path.join(SITE, `tools/glm_parity/parity_result_${VAR}.json`), JSON.stringify({ summary, rows }, null, 1));
