// Chat page check (headless Chromium, playwright): the answers shown in chat.html must equal the GLM engine's answers on
// the main page (index.html, window.glmDemo.ask) for the same questions on the general pack. Also takes desktop and
// phone (iPhone 13, 390 px) screenshots, light and dark, checks for console errors and horizontal scroll.
//   node chat_parity.mjs [base url] [screenshot dir] [out json]
// (run it from a directory where playwright is installed; copy the file there, as phone_general.mjs is run)
import { chromium, devices } from "playwright";
import fs from "fs";
const BASE = process.argv[2] || "http://localhost:8765/";
const SHOTS = process.argv[3] || ".";
const OUT = process.argv[4];
const QS = [
  "When did the author of Pride and Prejudice die?",
  "Who was the tallest US president ever?",
  "Is Lake Michigan or Lake Superior larger by surface area?",
  "How many members of Blackpink are there?",
  "Who directed the film Inception?",
  "What is the capital of Australia?",
  "zzqx blorf wibble?",
];
fs.mkdirSync(SHOTS, { recursive: true });
const browser = await chromium.launch();
const errs = [];
const watch = (page, tag) => {
  page.on("console", m => { if (m.type() === "error") errs.push(`${tag}: ${m.text()}`); });
  page.on("pageerror", e => errs.push(`${tag} pageerror: ${e.message}`));
};

// 1. reference: the engine on the main page
const refCtx = await browser.newContext();
const ref = await refCtx.newPage(); watch(ref, "index");
await ref.goto(BASE + "index.html?pack=general");
await ref.waitForFunction(() => window.glmDemo && window.glmDemo.ready && window.glmDemo.packInfo("general"), null, { timeout: 240000 });
const refAns = await ref.evaluate(async qs => {
  const out = [];
  for (const q of qs) { const r = await window.glmDemo.ask(q, "general"); out.push(r.ok ? (r.answers || []).map(String) : []); }
  return out;
}, QS);
await refCtx.close();

// 2. the chat page, desktop, asking through the UI (Enter in the composer)
async function chatRun(ctxOpts, tag, ask = true) {
  const ctx = await browser.newContext(ctxOpts);
  const page = await ctx.newPage(); watch(page, tag);
  await page.goto(BASE + "chat.html");
  await page.waitForFunction(() => window.glmChat && window.glmChat.ready, null, { timeout: 240000 });
  await page.screenshot({ path: `${SHOTS}/${tag}_01_empty.png` });
  const rows = [];
  if (ask) for (const [i, q] of QS.entries()) {
    const t0 = Date.now();
    await page.fill("#q", q);
    await page.press("#q", "Enter");
    await page.waitForFunction(n => document.querySelectorAll(".msg.glm[data-i]").length === n && window.glmChat.ready, i + 1, { timeout: 180000 });
    const last = page.locator(".msg.glm[data-i]").last();
    const shown = await last.locator(".a").evaluateAll(a => a.map(x => x.dataset.raw));
    const idk = await last.locator(".idk").count() > 0;
    const engine = await page.evaluate(async q => { const r = await window.glmChat.engineAsk(q, "general"); return r.ok ? (r.answers || []).map(String) : []; }, q);
    rows.push({ q, chat: shown, idk, index_engine: refAns[i], chat_engine: engine,
      same: JSON.stringify(shown.slice(0, 8)) === JSON.stringify(refAns[i].slice(0, 8)) && JSON.stringify(engine) === JSON.stringify(refAns[i]), wall_ms: Date.now() - t0 });
  }
  return { ctx, page, rows };
}
const desk = await chatRun({ viewport: { width: 1280, height: 900 }, colorScheme: "light" }, "desktop");
await desk.page.locator("details.how").first().evaluate(d => { d.open = true; }); await new Promise(r => setTimeout(r, 600));
await desk.page.locator(".msg.glm[data-i]").first().scrollIntoViewIfNeeded();
await desk.page.screenshot({ path: `${SHOTS}/desktop_02_chat_how.png` });
await desk.page.locator(".msg.glm[data-i]").last().scrollIntoViewIfNeeded();
await desk.page.screenshot({ path: `${SHOTS}/desktop_03_chat_end.png` });
// a past chat survives a reload (localStorage) and New chat shows the starters
await desk.page.reload();
await desk.page.waitForFunction(() => window.glmChat && window.glmChat.ready, null, { timeout: 240000 });
const histN = await desk.page.locator("#chats li").count();
await desk.ctx.close();

const darkDesk = await chatRun({ viewport: { width: 1280, height: 900 }, colorScheme: "dark" }, "desktop_dark", false);
await darkDesk.page.click(".starter >> nth=0");
await darkDesk.page.waitForFunction(() => document.querySelector(".msg.glm[data-i]") && window.glmChat.ready, null, { timeout: 180000 });
await darkDesk.page.locator("details.how").first().evaluate(d => { d.open = true; }); await new Promise(r => setTimeout(r, 600));
await darkDesk.page.screenshot({ path: `${SHOTS}/desktop_dark_02_answer.png` });
await darkDesk.ctx.close();

// 3. phone
const phone = await chatRun({ ...devices["iPhone 13"], colorScheme: "light" }, "iphone13", false);
const p = phone.page;

await p.click(".starter >> nth=1");
await p.waitForFunction(() => document.querySelector(".msg.glm[data-i]") && window.glmChat.ready, null, { timeout: 180000 });
await p.locator("details.how").first().evaluate(d => { d.open = true; }); await new Promise(r => setTimeout(r, 600));
await p.screenshot({ path: `${SHOTS}/iphone13_02_answer.png` });
await p.fill("#q", "zzqx blorf wibble?"); await p.press("#q", "Enter");
await p.waitForFunction(() => document.querySelectorAll(".msg.glm[data-i]").length === 2 && window.glmChat.ready, null, { timeout: 180000 });
await p.waitForTimeout(600); await p.screenshot({ path: `${SHOTS}/iphone13_03_idk.png` });
const sw = await p.evaluate(() => [document.documentElement.scrollWidth, document.querySelector(".scroll").scrollWidth, document.querySelector(".scroll").clientWidth]);
await p.click("#open-side"); await p.waitForTimeout(400);
await p.screenshot({ path: `${SHOTS}/iphone13_04_drawer.png` });
const focusInDrawer = await p.evaluate(() => document.getElementById("side").contains(document.activeElement));
await p.keyboard.press("Escape"); await p.waitForTimeout(300);
const drawerClosed = await p.evaluate(() => !document.getElementById("app").classList.contains("drawer"));
await phone.ctx.close();

const out = { base: BASE, pack: "general", n: desk.rows.length, same: desk.rows.filter(r => r.same).length, rows: desk.rows,
  history_after_reload: histN, phone_no_hscroll: sw[0] <= 390 && sw[1] <= sw[2], phone_scroll_widths: sw, drawer_focus: focusInDrawer, drawer_escape_closes: drawerClosed,
  console_errors: errs };
console.log(JSON.stringify(out, null, 1));
if (OUT) fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
await browser.close();
