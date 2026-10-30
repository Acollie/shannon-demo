// node parity.mjs <ref.json> <variant: fp32|...> [modelsDir]
import fs from "fs";
import * as ort from "onnxruntime-web";
const SITE = "/Users/alexcollie/Documents/code/github/shannon-demo";
const { Engine } = await import(SITE + "/assets/qa/engine.js");
const { ShannonQA } = await import(SITE + "/assets/qa/shannon_qa.js");
const [refPath, variant = "fp32", mdir = new URL("./models/", import.meta.url).pathname] = process.argv.slice(2);
ort.env.wasm.numThreads = 1;
const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));
const ex = JSON.parse(fs.readFileSync(SITE + "/qa/examples.json", "utf8"));
const extra = fs.existsSync(SITE + "/qa/extra_docs.json") ? JSON.parse(fs.readFileSync(SITE + "/qa/extra_docs.json", "utf8")).docs : [];
const vfile = n => { const v = variant.includes(":") ? Object.fromEntries(variant.split(",").map(x => x.split(":")))[n] : variant; return v === "shipped" ? `${SITE}/models/qa/${n}.onnx` : `${mdir}/${n}.${v}.onnx`; };
const uv = JSON.parse(fs.readFileSync(SITE + "/models/qa/yesno_vocab.json", "utf8"));
const eng = new Engine(ort, { embed: vfile("embed"), reader: vfile("reader"), yesno: vfile("yesno"),
  vocab: fs.readFileSync(SITE + "/models/qa/vocab.txt", "utf8"), unigram: uv });
await eng.init(async f => new Uint8Array(fs.readFileSync(f)));
const sys = new ShannonQA(eng, JSON.parse(fs.readFileSync(SITE + "/models/qa/gate.json", "utf8")));
const norm = a => JSON.stringify(a);
let same = 0, n = 0, sameForm = 0;
const err = { p_yes0: 0, qa_best: 0, cos_max: 0, margin: 0 };
const rows = [], times = [];
for (const d of ref) {
  const doc = [...ex.docs, ...extra].find(x => x.id === d.id);
  const t0 = performance.now();
  await sys.read(fs.readFileSync(SITE + "/" + doc.file, "utf8"));
  const tIndex = performance.now() - t0;
  for (const q of d.questions) {
    const r = await sys.ask(q.q);
    times.push(r.ms);
    const pyB = q.cands.find(c => c.form === "bool" && c.answer === true).feats, jsB = r.cands.find(c => c.form === "bool" && c.answer === true).feats;
    for (const k of ["p_yes0", "qa_best", "cos_max"]) err[k] = Math.max(err[k], Math.abs(pyB[k] - jsB[k]));
    if (norm(r.answer) === norm(q.answer)) err.margin = Math.max(err.margin, Math.abs(r.margin - q.margin));
    n++;
    const ok = norm(r.answer) === norm(q.answer) && r.form === q.form;
    same += ok; sameForm += r.form === q.form;
    const pk = norm(r.retrieval.picked) === norm(q.picked);
    rows.push({ doc: d.id, q: q.q, py: [q.form, q.answer, +q.margin.toFixed(3)], js: [r.form, r.answer, +r.margin.toFixed(3)], ok, picked_same: pk });
    if (!ok) {
      console.log("DIFF", d.id, q.q, "\n  py", q.form, norm(q.answer), q.margin.toFixed(3), q.picked, "\n  js", r.form, norm(r.answer), r.margin.toFixed(3), r.retrieval.picked);
      // compare candidate lists
      const pc = q.cands.map(c => c.form + "|" + norm(c.answer)), jc = r.cands.map(c => c.form + "|" + norm(c.answer));
      console.log("  cands same:", norm(pc) === norm(jc), pc.length, jc.length);
    }
  }
  console.log(d.id, "index ms", tIndex.toFixed(0));
}
const ts = times.sort((a, b) => a - b);
console.log(JSON.stringify({ variant, n, same, sameForm, rate: same / n, median_ms: ts[Math.floor(ts.length / 2)], max_ms: ts[ts.length - 1], max_abs_err: err }));
fs.writeFileSync(new URL(`./parity_${variant.replace(/[:,]/g, "_")}.json`, import.meta.url), JSON.stringify(rows, null, 1));
