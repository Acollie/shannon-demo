// Phone-sized test of the general pack: iPhone 13 emulation (390x844 @3x, touch, mobile UA), cold cache.
import { chromium, devices } from "playwright";
import fs from "fs";
const URL = process.argv[2] || "http://localhost:8765/";
const SHOTS = process.argv[3];
const QS = JSON.parse(fs.readFileSync(process.argv[4])).examples.map(e => e.q);
const device = process.argv[5] === "desktop" ? { viewport: { width: 1280, height: 900 } } : devices["iPhone 13"];
const browser = await chromium.launch();
const ctx = await browser.newContext({ ...device, colorScheme: process.argv[6] || "light" });
const page = await ctx.newPage();
const cdp = await ctx.newCDPSession(page);
await cdp.send("Performance.enable");
await cdp.send("Network.enable");
let wire = 0; cdp.on("Network.loadingFinished", e => { wire += e.encodedDataLength; });
if (process.env.THROTTLE) {   // approximate a mid-range phone: 4x slower CPU, ~"fast 4G" network (20 Mbit/s down, 40 ms RTT)
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 40, downloadThroughput: 20e6 / 8, uploadThroughput: 5e6 / 8 });
}
const errs = [];
page.on("console", m => { if (m.type() === "error") errs.push(m.text()); });
page.on("pageerror", e => errs.push("pageerror " + e.message));
let bytes = 0, reqs = 0;
page.on("response", async r => { try { const b = await r.body(); bytes += b.length; reqs++; } catch {} });
const mem = async () => { const m = (await cdp.send("Performance.getMetrics")).metrics; const g = k => (m.find(x => x.name === k) || {}).value || 0; return { js_heap_mb: +(g("JSHeapUsedSize") / 1048576).toFixed(1), js_heap_total_mb: +(g("JSHeapTotalSize") / 1048576).toFixed(1) }; };
const t0 = Date.now();
await page.goto(URL);
await page.waitForFunction(() => window.glmDemo && window.glmDemo.ready, null, { timeout: 180000 });
const tReady = Date.now() - t0, bReady = bytes, wReady = wire;
const tag = (process.argv[5] === "desktop" ? "desktop" : "iphone13") + (process.env.THROTTLE ? "_throttled" : "");
await page.screenshot({ path: `${SHOTS}/${tag}_01_ready_movies.png`, fullPage: false });
const t1 = Date.now();
await page.click('[data-pick="general"]');
await page.waitForFunction(() => { const p = window.glmDemo.packInfo("general"); return p && p !== "loading" && !document.getElementById("go").disabled; }, null, { timeout: 120000 });
const tGen = Date.now() - t1, bGen = bytes - bReady, wGen = wire - wReady;
await page.screenshot({ path: `${SHOTS}/${tag}_02_general_picked.png`, fullPage: false });
const out = { device: tag, ready_movies_s: tReady / 1000, ready_bytes_mb: +(bReady / 1e6).toFixed(1), general_core_s: tGen / 1000, general_core_mb: +(bGen / 1e6).toFixed(2), general_core_wire_mb: +(wGen / 1e6).toFixed(2), ready_wire_mb: +(wReady / 1e6).toFixed(1), questions: [] };
for (let i = 0; i < QS.length; i++) {
  const b0 = bytes, w0 = wire, ts = Date.now();
  await page.click(`.ex >> nth=${i}`);
  await page.waitForFunction(q => { const ql = document.querySelector(".qline"); return ql && ql.textContent.includes(q.slice(0, 20)) && !document.getElementById("go").disabled; }, QS[i], { timeout: 180000 });
  const dt = Date.now() - ts;
  const ans = await page.$$eval(".answers .a", a => a.map(x => x.textContent));
  const path = await page.$eval(".sentence", e => e.textContent).catch(() => "");
  const hasGraph = await page.$("#gc") !== null;
  out.questions.push({ wire_mb: +((wire - w0) / 1e6).toFixed(2), q: QS[i], answer: ans, path, ms_wall: dt, mb: +((bytes - b0) / 1e6).toFixed(2), graph: hasGraph, ...(await mem()) });
  if (i === 0 || i === 4) {
    await page.waitForTimeout(3000);
    await page.locator(".result").screenshot({ path: `${SHOTS}/${tag}_q${i + 1}_result.png` });
  }
  if (i === 0) {   // tap an answer dot: facts load
    const pt = await page.evaluate(() => { const g = window.glmDemo.graph(); const n = g.viz.nodes.find(n => n.answer) || g.viz.nodes[1]; const r = document.getElementById("gc").getBoundingClientRect(); return [r.left + n.x, r.top + n.y + window.scrollY, n.name]; });
    await page.evaluate(([x, y]) => window.scrollTo(0, 0), pt);
    const box = await page.$eval("#gc", e => { const r = e.getBoundingClientRect(); return [r.left, r.top]; });
    const nn = await page.evaluate(() => { const g = window.glmDemo.graph(); const n = g.viz.nodes.find(n => n.answer) || g.viz.nodes[1]; return [n.x, n.y, n.name]; });
    await page.locator("#gc").click({ position: { x: nn[0], y: nn[1] } });
    await page.waitForFunction(() => !/Looking up|Tap a dot/.test(document.getElementById("gfacts").textContent), null, { timeout: 30000 });
    out.facts_tap = { node: nn[2], text: (await page.textContent("#gfacts")).slice(0, 160) };
    await page.locator(".think").screenshot({ path: `${SHOTS}/${tag}_q1_facts.png` });
  }
}
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
out.no_hscroll = sw <= (device.viewport.width);
out.total_mb = +(bytes / 1e6).toFixed(1); out.requests = reqs;
out.mem_end = await mem();
out.errors = errs.slice(0, 10);
// movies still works after switching back
await page.click('[data-pick="movies"]');
await page.click(`.ex >> nth=0`);
await page.waitForFunction(() => document.querySelector(".qline") && document.querySelector(".qline").textContent.includes("Memento") && !document.getElementById("go").disabled, null, { timeout: 60000 });
out.movies_after = await page.$$eval(".answers .a", a => a.map(x => x.textContent));
await page.screenshot({ path: `${SHOTS}/${tag}_09_movies_again.png`, fullPage: false });
console.log(JSON.stringify(out, null, 1));
await browser.close();
