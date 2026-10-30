// GLM Pocket check (Playwright, FULL Chromium in new headless mode so WebGPU is real; headless-shell only has SwiftShader).
//   node pocket_check.mjs [base url] [screenshot dir] [out json] [profile dir]
// (run from a directory where playwright is installed, as chat_parity.mjs is)
// 1. graph-only answers in pocket.html == the main page engine's answers (5 questions)
// 2. About me: add / delete, persists across reload, "Forget everything" clears it (also after reload)
// 3. enable Qwen2.5-0.5B (WebGPU), 3 chat prompts + 3 fact questions: tier badges and routing match the adapter's rule
// 4. network: every request goes to the site itself or a static host (fonts, jsDelivr, Hugging Face, GitHub raw); GET only
// 5. 390 px phone layout: no horizontal scroll; no console errors anywhere
import { chromium, devices } from "playwright";
import fs from "fs";
const BASE = process.argv[2] || "http://localhost:8765/";
const SHOTS = process.argv[3] || ".";
const OUT = process.argv[4];
const PROF = process.argv[5] || "/tmp/pocket_profile";
fs.mkdirSync(SHOTS, { recursive: true });
const GRAPH_QS = [
  "When did the author of Pride and Prejudice die?",
  "Who wrote the Dark Tower series?",
  "Is Lake Michigan or Lake Superior larger by surface area?",
  "How many members of Blackpink are there?",
  "What is the capital of Australia?",
];
const CHAT_QS = ["Hi! How are you today?", "Give me three ideas for a quick weeknight dinner.", "Write a two-line poem about autumn rain."];
const FACT_QS = ["Who wrote the Dark Tower series?", "How many members of Blackpink are there?", "Who directed the film Inception?"];
const STATIC_HOSTS = [/^localhost$/, /^127\.0\.0\.1$/, /^acollie\.github\.io$/, /^fonts\.googleapis\.com$/, /^fonts\.gstatic\.com$/,
  /^cdn\.jsdelivr\.net$/, /^huggingface\.co$/, /(^|\.)hf\.co$/, /(^|\.)huggingface\.co$/, /^raw\.githubusercontent\.com$/];
const res = { base: BASE, checks: {} };
const errs = [], reqs = [];
const watch = (page, tag) => {
  page.on("console", m => { if (m.type() === "error") errs.push(`${tag}: ${m.text()}`); });
  page.on("pageerror", e => errs.push(`${tag} pageerror: ${e.message}`));
  page.on("request", r => reqs.push({ tag, method: r.method(), url: r.url() }));
  page.on("response", r => { if (r.status() >= 400) errs.push(`${tag} HTTP ${r.status()}: ${r.url()}`); });
  page.on("worker", w => { reqs.push({ tag, method: "WORKER", url: w.url() }); });
};
const ctx = await chromium.launchPersistentContext(PROF, { channel: "chromium", headless: true, args: ["--enable-unsafe-webgpu"], viewport: { width: 1280, height: 860 } });
ctx.on("request", r => reqs.push({ tag: "ctx", method: r.method(), url: r.url() }));   // includes worker requests
const page = ctx.pages()[0] || await ctx.newPage(); watch(page, "pocket");
page.on("dialog", d => d.accept());

// 0. reference answers from the main page engine (own tab, so index.html's console is tagged "index")
const refPage = await ctx.newPage(); watch(refPage, "index");
await refPage.goto(BASE + "index.html?pack=general", { waitUntil: "domcontentloaded", timeout: 120000 });
await refPage.waitForFunction(() => window.glmDemo && window.glmDemo.ready && window.glmDemo.packInfo("general"), null, { timeout: 240000 });
const refAns = await refPage.evaluate(async qs => { const o = []; for (const q of qs) { const r = await window.glmDemo.ask(q, "general"); o.push(r.ok ? (r.answers || []).map(String) : []); } return o; }, GRAPH_QS);

await refPage.close();
// fresh pocket state
await page.goto(BASE + "pocket.html", { waitUntil: "domcontentloaded", timeout: 120000 });
await page.evaluate(() => { try { localStorage.removeItem("glm-pocket-chats-v1"); localStorage.removeItem("glm-pocket-about-me-v1"); localStorage.removeItem("glm-pocket-lm-v1"); } catch {} });
await page.reload();
const ready = () => page.waitForFunction(() => window.glmPocket && window.glmPocket.ready, null, { timeout: 240000 });
await ready();
res.webgpu = await page.evaluate(async () => { const a = navigator.gpu && await navigator.gpu.requestAdapter(); return a ? { ok: true, info: a.info ? `${a.info.vendor} / ${a.info.architecture}` : "" } : { ok: false }; });
await page.screenshot({ path: `${SHOTS}/01_desktop_empty.png` });

async function askUI(q) {
  const n = await page.locator(".msg.glm[data-i]").count();
  await page.fill("#q", q); await page.press("#q", "Enter");
  await page.waitForFunction(n => document.querySelectorAll(".msg.glm[data-i]").length === n + 1 && window.glmPocket.ready, n, { timeout: 300000 });
  const last = page.locator(".msg.glm[data-i]").last();
  const tier = await last.locator(".badge").getAttribute("data-tier");
  const shown = await last.locator(".a").evaluateAll(a => a.map(x => x.dataset.raw));
  const text = (await last.locator(".lmtext").count()) ? await last.locator(".lmtext").innerText() : "";
  const r = await page.evaluate(() => { const c = window.glmPocket.chat(), m = c.msgs[c.msgs.length - 1]; return { tier: m.r.tier, type: m.r.type, score: m.r.score, ok: m.r.ok, low: !!m.r.low, route: m.r.route, suggest: m.suggest || null }; });
  const how = await last.locator("details.how summary").innerText().catch(() => "");
  return { q, tier_badge: tier, tier: r.tier, type: r.type, score: r.score, low: r.low, shown, text: text.slice(0, 300), route: r.route, how, suggest: r.suggest };
}

// 1. graph-only parity
const g = [];
for (const [i, q] of GRAPH_QS.entries()) { const r = await askUI(q); g.push({ ...r, ref: refAns[i], same: JSON.stringify(r.shown.slice(0, 8)) === JSON.stringify(refAns[i].slice(0, 8)) && r.tier === "graph" }); }
res.checks.graph_only = { same: g.filter(x => x.same).length, n: g.length, all_graph_badge: g.every(x => x.tier_badge === "graph"), how_shown: g.every(x => x.how === "How I got this"), rows: g };
await page.locator(".msg.glm[data-i]").last().locator("details.how summary").click();
await page.screenshot({ path: `${SHOTS}/02_desktop_graph_only.png` });

// 2. About me
await page.click("#new");
for (const f of ["My name is Sam", "I'm vegetarian", "I live in Leeds"]) { await page.fill("#mem-in", f); await page.click("#mem-addbtn"); }
const before = await page.locator("#mem-list li").count();
await page.locator('#mem-list li:has-text("Leeds") [data-mdel]').click();
const afterDel = await page.locator("#mem-list li span").allInnerTexts();
const memQ = await askUI("What is my name?");     // graph-only: answered from About me
await page.reload(); await ready();
const afterReload = await page.locator("#mem-list li span").allInnerTexts();
res.checks.memory = { added: before, after_delete: afterDel, after_reload: afterReload, persists: JSON.stringify(afterDel) === JSON.stringify(afterReload) && afterReload.length === 2, graph_only_my_name: memQ };

// 3. enable the 0.5B chat model
const t0 = Date.now();
await page.click("#lm-card [data-lmon]");
await page.waitForTimeout(1500);
await page.screenshot({ path: `${SHOTS}/03_desktop_lm_loading.png` });
await page.waitForFunction(() => window.glmPocket.model.lmState === "ready" || window.glmPocket.model.lmState === "error", null, { timeout: 900000 });
res.lm_load_s = (Date.now() - t0) / 1000;
res.lm_state = await page.evaluate(() => ({ state: window.glmPocket.model.lmState, err: window.glmPocket.model.lmError, key: window.glmPocket.model.lmKey }));
await page.click("#new");
const chat = [], fact = [];
for (const q of CHAT_QS) chat.push(await askUI(q));
for (const q of FACT_QS) fact.push(await askUI(q));
const rule = await page.evaluate(() => ({ threshold: window.glmPocket.model.graphMin, types: window.glmPocket.model.cal.types }));
const expectTier = r => (r.type && rule.types.includes(r.type) && r.score != null && r.score >= rule.threshold ? "graph" : "lm");
res.checks.lm = { rule, chat_all_lm: chat.every(x => x.tier === "lm" && x.tier_badge === "lm" && x.text.length > 0),
  fact_routing_ok: fact.every(x => x.tier_badge === x.tier && (x.tier === "graph" ? expectTier(x) === "graph" : true)), chat, fact };
// About me used by the LM + save suggestion
const personal = await askUI("What should I cook for dinner tonight, given what you know about me?");
const stmt = await askUI("My favourite colour is green.");
let saved = false;
if (stmt.suggest) { await page.locator(".msg.glm[data-i]").last().locator("[data-save]").click(); saved = await page.locator('#mem-list li:has-text("green")').count() > 0; }
res.checks.lm_memory = { personal, statement: stmt, suggestion_saved: saved };
res.layout_after_lm = await page.evaluate(() => ({ html_scroll: document.documentElement.scrollTop, app_h: document.getElementById("app").getBoundingClientRect().height, vh: innerHeight }));
await page.screenshot({ path: `${SHOTS}/04_desktop_lm_chat.png`, fullPage: false });
await page.locator(".msg.glm[data-i]").nth(3).scrollIntoViewIfNeeded();
await page.screenshot({ path: `${SHOTS}/05_desktop_lm_fact.png` });

// forget everything, then reload
await page.click("#mem-forget");
const afterForget = await page.locator("#mem-list li").count();
await page.reload(); await ready();
const afterForgetReload = await page.locator("#mem-list li").count();
const stored = await page.evaluate(() => { try { return localStorage.getItem("glm-pocket-about-me-v1"); } catch { return "blocked"; } });
res.checks.forget = { after_forget: afterForget, after_reload: afterForgetReload, storage_key: stored, ok: afterForget === 0 && afterForgetReload === 0 && !stored };
await page.waitForFunction(() => window.glmPocket.model.lmState === "ready", null, { timeout: 120000 }).catch(() => {});
res.lm_reload_state = await page.evaluate(() => window.glmPocket.model.lmState);
await page.screenshot({ path: `${SHOTS}/06_desktop_after_forget.png` });
await ctx.close();

// 5. phone, 390 px (graph-only, fresh context)
const br = await chromium.launch({ channel: "chromium", headless: true });
for (const scheme of ["light", "dark"]) {
  const pc = await br.newContext({ ...devices["iPhone 13"], colorScheme: scheme });
  pc.on("request", r => reqs.push({ tag: "phone", method: r.method(), url: r.url() }));
  const p = await pc.newPage(); watch(p, "phone-" + scheme);
  await p.goto(BASE + "pocket.html", { waitUntil: "domcontentloaded", timeout: 120000 });
  await p.waitForFunction(() => window.glmPocket && window.glmPocket.ready, null, { timeout: 240000 });
  await p.screenshot({ path: `${SHOTS}/07_phone_${scheme}_empty.png` });
  await p.fill("#q", "Who wrote the Dark Tower series?"); await p.press("#q", "Enter");
  await p.waitForFunction(() => document.querySelectorAll(".msg.glm[data-i]").length === 1 && window.glmPocket.ready, null, { timeout: 120000 });
  await p.screenshot({ path: `${SHOTS}/08_phone_${scheme}_answer.png` });
  const sw = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth, b: document.body.scrollWidth }));
  await p.click("#open-side"); await p.waitForTimeout(400);
  await p.fill("#mem-in", "I'm left-handed"); await p.click("#mem-addbtn");
  await p.screenshot({ path: `${SHOTS}/09_phone_${scheme}_drawer.png` });
  const sw2 = await p.evaluate(() => document.documentElement.scrollWidth);
  res.checks["phone_" + scheme] = { ...sw, drawer_sw: sw2, no_hscroll: sw.sw <= sw.iw && sw.b <= sw.iw && sw2 <= sw.iw };
  await pc.close();
}
await br.close();

const hosts = {};
for (const r of reqs) { if (r.method === "WORKER") continue; try { const u = new URL(r.url); if (u.protocol === "data:" || u.protocol === "blob:") continue; hosts[u.hostname] = (hosts[u.hostname] || 0) + 1; } catch {} }
res.checks.network = { hosts, non_static: Object.keys(hosts).filter(h => !STATIC_HOSTS.some(rx => rx.test(h))),
  non_get: [...new Set(reqs.filter(r => r.method !== "GET" && r.method !== "WORKER").map(r => `${r.method} ${r.url}`))] };
res.console_errors = errs.filter(e => !e.startsWith("index"));
res.index_console_errors = errs.filter(e => e.startsWith("index"));
console.log(JSON.stringify(res, null, 1));
if (OUT) fs.writeFileSync(OUT, JSON.stringify(res, null, 1));
