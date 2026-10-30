// Same 200-question parity as parity_general.mjs, but inside headless Chromium on the real page (fp16 encoder).
import { chromium } from "playwright"; import fs from "fs";
const [URL, REF, OUT] = process.argv.slice(2);
const ref = JSON.parse(fs.readFileSync(REF)).rows;
const b = await chromium.launch(); const page = await (await b.newContext()).newPage();
await page.goto(URL); await page.waitForFunction(() => window.glmDemo && window.glmDemo.ready, null, { timeout: 180000 });
await page.evaluate(() => window.glmDemo.pick("general"));
await page.waitForFunction(() => { const p = window.glmDemo.packInfo("general"); return p && p !== "loading"; }, null, { timeout: 120000 });
const t0 = Date.now();
const res = await page.evaluate(async qs => { const out = []; for (const q of qs) { const o = await window.glmDemo.ask(q, "general"); out.push({ topics: o.topics, top: o.top, node: o.node, answer: o.answer }); } return out; }, ref.map(r => r.q));
let p = 0, a = 0, both = 0;
res.forEach((o, i) => { const r = ref[i]; const okP = JSON.stringify(o.top) === JSON.stringify(r.top[0] || null);
  const ja = o.node != null && o.node >= 0 ? o.node : o.answer, pa = r.node != null ? r.node : r.answer; const okA = JSON.stringify(ja) === JSON.stringify(pa);
  p += okP; a += okA; both += okP && okA; if (!(okP && okA)) console.log("DIFF", r.q); });
const s = { n: ref.length, same_top_path: p, same_answer: a, same_both: both, wall_s: (Date.now() - t0) / 1000, browser: "headless Chromium (playwright)", encoder: "fp16" };
console.log(JSON.stringify(s)); fs.writeFileSync(OUT, JSON.stringify(s, null, 1)); await b.close();
