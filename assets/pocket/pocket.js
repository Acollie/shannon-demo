// GLM Pocket page: UI only (layout and look shared with chat.html via assets/chat/chat.css). The model is the two-tier
// adapter in ./tiered.js: Shannon's graph first, then (if the user enabled it) a small LM on this device.
// Chats and "About me" facts live in this browser's localStorage; nothing is sent to a server.
import { TieredModel, LMS, intentOf } from "./tiered.js";
import { Memory, suggestFact } from "./memory.js";

const $ = id => document.getElementById(id);
const BASE = new URL("../../", import.meta.url).href;
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = v => (v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString("en-US")) + " ms";
const fmtBytes = b => (b < 1024 ? b + " B" : b < 1048576 ? Math.round(b / 1024) + " kB" : (b / 1048576).toFixed(1) + " MB");
const fmtName = s => (/^-?\d+(\.\d+)?$/.test(s) && Math.abs(+s) >= 10000 ? (+s).toLocaleString("en-US") : s);
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const KEY = "glm-pocket-chats-v1", PREF = "glm-pocket-lm-v1", MAX_CHATS = 40, SHOW = 8;
const desktop = () => matchMedia("(pointer: fine) and (min-width: 861px)").matches;

const model = new TieredModel(BASE);
const memory = new Memory();
const state = { ready: false, busy: false, active: "general", chats: [], cur: null, gpu: null, lmKey: null, lmProg: null };

// ------------------------------------------------------------------ storage (per-browser convenience; may be blocked)
function loadChats() { try { const v = JSON.parse(localStorage.getItem(KEY) || "[]");
  return Array.isArray(v) ? v.filter(c => c && Array.isArray(c.msgs)).map(c => ({ ...c, msgs: c.msgs.filter(m => !m.pending) })) : []; } catch { return []; } }
function saveChats() { try { localStorage.setItem(KEY, JSON.stringify(state.chats.slice(0, MAX_CHATS))); } catch { /* storage blocked or full */ } }
const getPref = () => { try { return JSON.parse(localStorage.getItem(PREF) || "null"); } catch { return null; } };
const setPref = v => { try { v ? localStorage.setItem(PREF, JSON.stringify(v)) : localStorage.removeItem(PREF); } catch { /* blocked */ } };
const curChat = () => state.chats.find(c => c.id === state.cur) || null;
const packTitle = id => { const p = model.catalog().find(x => x.id === id); return p ? (p.pick || p.title) : id; };

// ------------------------------------------------------------------ sidebar: chats
function renderChats() {
  $("chats").innerHTML = state.chats.map(c => `<li${c.id === state.cur ? ' aria-current="true"' : ""}>
    <button class="open" type="button" data-open="${esc(c.id)}" title="${esc(c.title)}">${esc(c.title)}</button>
    <button class="del" type="button" data-del="${esc(c.id)}" aria-label="Delete chat: ${esc(c.title)}">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M10 11v6M14 11v6M6 7l1 12h10l1-12M9 7V4h6v3"/></svg></button></li>`).join("");
  $("no-chats").hidden = state.chats.length > 0;
  const c = curChat();
  $("top-pack").textContent = c ? c.title : "GLM Pocket";
}
$("chats").addEventListener("click", e => {
  const o = e.target.closest("[data-open]"), d = e.target.closest("[data-del]");
  if (o) { state.cur = o.dataset.open; renderAll(); closeDrawer(); scrollEnd(false); $("q").focus(); }
  if (d) {
    state.chats = state.chats.filter(c => c.id !== d.dataset.del);
    if (state.cur === d.dataset.del) state.cur = null;
    saveChats(); renderAll();
  }
});
$("new").addEventListener("click", () => { state.cur = null; renderAll(); closeDrawer(); $("q").focus(); });

// ------------------------------------------------------------------ sidebar: About me
function renderMemory() {
  const items = memory.list();
  $("mem-list").innerHTML = items.map(x => `<li><span>${esc(x.text)}</span>
    <button class="del" type="button" data-mdel="${esc(x.id)}" aria-label="Delete: ${esc(x.text)}">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg></button></li>`).join("");
  $("mem-empty").hidden = items.length > 0;
  $("mem-forget").hidden = items.length === 0;
  $("mem-where").textContent = memory.durable ? "Stored only in this browser. Nothing is sent anywhere."
    : "This browser blocks storage, so these facts last only until you close the tab. Nothing is sent anywhere.";
}
$("mem-form").addEventListener("submit", e => { e.preventDefault(); if (memory.add($("mem-in").value)) { $("mem-in").value = ""; renderMemory(); } });
$("mem-list").addEventListener("click", e => { const d = e.target.closest("[data-mdel]"); if (d) { memory.remove(d.dataset.mdel); renderMemory(); } });
$("mem-forget").addEventListener("click", () => {
  if (!confirm("Forget everything in About me? This removes all saved facts from this browser.")) return;
  memory.forget(); renderMemory();
});

// ------------------------------------------------------------------ chat model (opt-in, remembered once accepted)
const lmSize = key => TieredModel.sizeMB(key, state.gpu);
const lmLabel = key => LMS[key].label;
// the one-tap offer, used in the banner, the sidebar and "can't answer" replies
function lmOffer(compact = false) {
  const g = state.gpu, s = model.lmState, key = state.lmKey;
  if (!g || !key) return `<p class="lm-off">Checking what this device can run…</p>`;
  if (s === "ready") return `<p class="lm-on"><span class="dot" aria-hidden="true"></span>The chat model (${esc(lmLabel(model.lmKey))}) is on. It runs on this device; nothing you type is sent anywhere.</p>`;
  if (s === "loading") {
    const p = state.lmProg;
    return `<p class="lm-off">Turning on the chat model (${esc(lmLabel(model.lmKey || key))})… ${compact ? "Ask again when it's ready, or just send: your message waits for it." : "The download happens once; after that it loads from this browser's cache."}</p>
      <div class="track lmtrack"><i style="width:${p && p.frac != null ? Math.round(p.frac * 100) : 2}%"></i></div>
      ${compact ? "" : `<p class="lm-prog">${esc(p ? p.text : "Starting…")}</p>`}`;
  }
  const err = s === "error" ? `<p class="lm-err">The chat model did not load: ${esc(model.lmError)}</p>` : "";
  const note = g.note ? `<p class="lm-note">${esc(g.note)}</p>` : "";
  const alt = compact || LMS[key].engine !== "webllm" || g.phone ? "" : key === "large"
    ? `<button class="linkbtn" type="button" data-lmkey="small">Use the smaller model instead (${LMS.small.mb} MB, faster, weaker)</button>`
    : `<button class="linkbtn" type="button" data-lmkey="large">Use the better model instead (${LMS.large.mb} MB)</button>`;
  return `${err}<button class="lm-enable" type="button" data-lmon ${state.ready ? "" : "disabled"}>Turn on the chat model (${lmSize(key)} MB, stays on your device)</button>
    ${compact ? note : `<p class="lm-note">${esc(lmLabel(key))}. Downloaded once from Hugging Face, then loaded from this browser's cache. Without it, GLM only answers clear fact questions from its knowledge graph.</p>${note}${alt}`}`;
}
function renderLM() {
  $("lm-side").innerHTML = lmOffer() + (model.lmState === "ready" ? `<button class="btn" type="button" data-lmoff>Turn off</button>` : "");
  $("lm-card").innerHTML = model.lmState === "ready" ? "" : lmOffer();
  $("lm-card").hidden = model.lmState === "ready";
  const p = $("lm-pill");
  p.textContent = model.lmReady() ? `Graph + ${lmLabel(model.lmKey)}` : model.lmState === "loading" ? "Loading chat model…" : "Graph only";
  p.dataset.on = model.lmReady();
  for (const el of document.querySelectorAll("[data-offer]")) el.innerHTML = lmOffer(true);
}
document.addEventListener("click", e => {
  if (e.target.closest("[data-lmon]")) enableLM(state.lmKey);
  const k = e.target.closest("[data-lmkey]");
  if (k) { state.lmKey = k.dataset.lmkey; renderLM(); }
  if (e.target.closest("[data-lmoff]")) { setPref(null); location.reload(); }
});
async function enableLM(key) {
  if (model.lmState === "loading" || model.lmReady()) return;
  state.lmKey = key; state.lmProg = null;
  setPref({ key });   // consent is remembered once given: later visits load it from the cache without asking
  const t = model.enableLM(key, p => { state.lmProg = p; renderLM(); });
  renderLM(); renderComposer(); if (!state.busy) renderThread();
  try { await t; state.lmKey = model.lmKey; }
  catch (e) { console.warn("[pocket] chat model failed to load", e); setPref(null); }
  renderLM(); renderComposer(); if (!state.busy) renderThread();
}

// ------------------------------------------------------------------ sidebar: knowledge packs (same as chat.html)
function renderPacks() {
  const cat = model.catalog().slice().sort((a, b) => (b.id === "general") - (a.id === "general"));
  $("packs").innerHTML = cat.map(p => {
    const s = model.packState(p.id), avail = p.status === "available";
    const loading = s === "loading", live = s && !loading;
    const tag = !avail ? `<span class="st off">coming next</span>` : loading ? `<span class="st off">loading…</span>` : live ? `<span class="st">loaded</span>` : `<span class="st off">not loaded</span>`;
    const size = live ? (s.lazy ? `${fmtBytes(s.bytes)} loaded; facts download as you ask (${fmtBytes(s.total_bytes)} in all)` : `${fmtBytes(s.bytes)} loaded`) : (p.bytes ? fmtBytes(p.bytes) : "");
    const btn = !avail ? "" : live ? `<button class="btn act" type="button" data-unload="${p.id}">Unload</button>`
      : `<button class="btn act" type="button" data-load="${p.id}" ${state.ready && !loading ? "" : "disabled"}>${loading ? "Loading…" : "Load"}</button>`;
    return `<div class="pk" data-active="${p.id === state.active}">
      <input type="radio" name="pk" id="pk-${p.id}" value="${p.id}" ${p.id === state.active ? "checked" : ""} ${avail ? "" : "disabled"} aria-describedby="pkd-${p.id}">
      <label for="pk-${p.id}">${esc(p.pick || p.title)}</label>${tag}
      <div class="d" id="pkd-${p.id}">${esc(p.detail)}${size ? ` · ${esc(size)}` : ""}</div>${btn}</div>`;
  }).join("");
}
$("packs").addEventListener("change", e => { if (e.target.name === "pk") pick(e.target.value); });
$("packs").addEventListener("click", e => {
  const l = e.target.closest("[data-load]"), u = e.target.closest("[data-unload]");
  if (l) loadPack(l.dataset.load);
  if (u) { model.unloadPack(u.dataset.unload); renderAll(); }
});
async function pick(id) {
  state.active = id;
  try { const u = new URL(location.href); u.searchParams.set("pack", id); history.replaceState(null, "", u); } catch { /* sandboxed */ }
  renderAll();
  if (state.ready && !model.packState(id)) await loadPack(id);
}
async function loadPack(id) {
  showLoad(`Loading the ${packTitle(id)} pack…`, null);
  renderAll();
  try { await model.loadPack(id, p => showLoad(p.text, p.frac)); hideLoad(); }
  catch (e) { console.error("[pocket] pack load failed", e); showLoad(`The ${packTitle(id)} pack did not load: ${e.message || e}`, null, true); }
  renderAll();
}

// ------------------------------------------------------------------ load bar + composer state
function showLoad(text, frac, bad = false) {
  $("loadbar").hidden = false; $("loadbar").className = "loadbar" + (bad ? " bad" : "");
  $("load-t").textContent = text;
  $("load-track").hidden = frac == null;
  if (frac != null) $("load-i").style.width = Math.round(frac * 100) + "%";
}
function hideLoad() { $("loadbar").hidden = true; }
function canAsk() { const s = model.packState(state.active); return state.ready && !state.busy && !!s && s !== "loading"; }
function renderComposer() {
  const s = model.packState(state.active), q = $("q");
  q.disabled = !state.ready;
  q.placeholder = !state.ready ? "Loading…" : s === "loading" ? `Loading the ${packTitle(state.active)} pack…`
    : !s ? `Load the ${packTitle(state.active)} pack to ask (menu)` : model.lmReady() ? "Message GLM Pocket…" : "Ask a fact question, or turn on the chat model…";
  $("send").disabled = !canAsk() || !q.value.trim();
  for (const b of document.querySelectorAll(".starter")) b.disabled = !canAsk();
}

// ------------------------------------------------------------------ thread
function chainHTML(r) {
  if (!r.path) return "";
  const parts = [`<span class="node topic">${esc(fmtName(r.path.from))}</span>`];
  r.path.steps.forEach((s, i) => {
    const last = i === r.path.steps.length - 1;
    const sample = s.sample.slice(0, 3).map(fmtName).join(", ") + (s.n > 3 ? ", …" : "");
    parts.push(`<span class="rel">${esc(s.relation)}</span>`);
    parts.push(`<span class="node${last ? " end" : ""}">${s.n === 1 ? esc(sample) : `<small>${s.n.toLocaleString("en-US")} reached</small>${esc(sample)}`}</span>`);
  });
  return `<div class="chain" aria-label="Graph path">${parts.join("")}</div>`;
}
const pathText = r => r.path ? [r.path.from, ...r.path.steps.flatMap(s => [s.relation, s.n === 1 ? s.sample[0] : `(${s.n} reached)`])].map(fmtName).join(" → ") : "";
// LM text: paragraphs, line breaks, **bold** and *italic* (everything else stays plain, escaped text)
const inline = t => t.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>");
const prose = t => esc(t).split(/\n{2,}/).map(p => `<p>${inline(p).replace(/\n/g, "<br>")}</p>`).join("");
const BADGE = {
  graph: `<span class="badge graph" data-tier="graph">Graph (path shown)</span>`,
  lm: `<span class="badge lm" data-tier="lm">Language model — may be wrong</span>`,
  memory: `<span class="badge mem" data-tier="memory">About me (saved on this device)</span>`,
  notice: `<span class="badge note" data-tier="notice">No answer</span>`,
};

function replyHTML(m, reveal) {
  const r = m.r, pk = packTitle(r.pack || m.pack), tier = r.tier || "graph";
  let head;
  if (m.nopack) head = `<p class="idk">The ${esc(pk)} pack is not loaded, so there is no graph to answer from.</p><p class="idk-why">Load it under Knowledge packs (menu), then ask again.</p>`;
  else if (tier === "lm") head = `<div class="lmtext">${prose(r.text || "…")}</div>`;
  else if (tier === "notice") {
    const gt = r.graph_tried;
    head = `<p class="idk">${r.intent === "fact" ? "I'm not sure enough to answer that from the knowledge graph." : "I need the chat model for that."}</p>
      <p class="idk-why">Without the chat model, GLM only answers clear fact questions (like “Who wrote Dracula?”) when its knowledge graph is confident.</p>
      <div class="offer" data-offer>${lmOffer(true)}</div>`
      + (gt && gt.ok ? `<details class="guess"><summary>Graph's best guess (below its confidence rule, so probably wrong)</summary><div class="guess-body">
        <div class="answers">${gt.answers.slice(0, SHOW).map(a => `<span class="a dim">${esc(fmtName(a))}</span>`).join("")}</div>${chainHTML(gt)}</div></details>` : "");
  }
  else if (tier === "memory") head = r.ok ? `<div class="lab">From your About me</div><ul class="memhit">${r.facts_used.map(f => `<li>${esc(f.text)}</li>`).join("")}</ul>`
    : `<p class="idk">You haven't saved anything about yourself yet.</p><p class="idk-why">Add facts under About me (menu), or turn on the chat model.</p>`;
  else if (r.ok) {
    const shown = r.answers.slice(0, SHOW).map(a => `<span class="a" data-raw="${esc(a)}">${esc(fmtName(a))}</span>`).join("");
    const more = r.answers.length > SHOW ? `<span class="more">+${r.answers.length - SHOW} more</span>` : "";
    head = `<div class="lab">Answer</div><div class="answers">${shown}${more}</div>`
      + (r.low ? `<p class="lowconf">Low confidence: this answer is below the graph's confidence rule, and answers like it are usually wrong. Treat it as a guess and check the path.</p>` : "")
      + (r.check ? `<p class="prec">The chat model, asked on its own, gave the same answer.</p>`
        : model.cal && r.pack === model.cal.pack ? `<p class="prec">Graph answers that pass its confidence rule were right about ${model.cal.precision_pct}% of the time in testing. Check the path below.</p>` : "");
  } else head = `<p class="idk">I don't know from this graph.</p>${r.reason ? `<p class="idk-why">${esc(r.reason)}</p>` : ""}${model.lmReady() ? "" : `<p class="idk-why">Turn on the chat model to talk about anything else.</p>`}`;

  const rows = [];
  if (r.route) rows.push(["Why this tier", esc(r.route)]);
  if (tier === "graph") {
    if (r.type) rows.push(["Answer type", esc(r.type) + (r.explain ? ` · ${esc(r.explain)}` : "")]);
    if (r.names && r.names.length) rows.push(["Names found", [...new Set(r.names)].map(n => esc(fmtName(n))).join(", ")]);
    if (r.score != null) rows.push(["Path score", `<span class="mono">${r.score.toFixed(2)}</span> (raw, not a probability)`]);
    if (r.check) rows.push(["Chat model check", `agrees: “${esc(r.check.text.slice(0, 120))}”`]);
  }
  if (tier === "lm") {
    rows.push(["Model", `${esc(r.model)} on this device`]);
    if (r.graph_facts && r.graph_facts.length) rows.push(["Graph facts given", r.graph_facts.map(esc).join("<br>") + " <small>(labelled “may be wrong”)</small>"]);
    const gt = r.graph_tried;
    if (gt) rows.push(["Graph's guess", gt.ok ? `${gt.answers.slice(0, 5).map(a => esc(fmtName(a))).join(", ")} (score <span class="mono">${gt.score != null ? gt.score.toFixed(2) : "n/a"}</span>)${r.graph_facts && r.graph_facts.length ? "; its candidate facts were given to the model, marked “may be wrong”" : ""}` : "none"]);
    if (gt && gt.disagree) rows.push(["Why not the graph", `its answer passed the confidence rule, but the chat model, asked on its own, said “${esc(gt.disagree.slice(0, 120))}”`]);
    if (r.facts_used && r.facts_used.length) rows.push(["About me used", r.facts_used.map(f => esc(f.text)).join("; ")]);
    if (r.first_token_ms) rows.push(["First word", `<span class="mono">${fmtMs(r.first_token_ms)}</span>`]);
  }
  if (r.ms) rows.push(["Time", `<span class="mono">${fmtMs(r.ms)}</span> on this device`]);
  if (r.bytes) rows.push(["Downloaded", `<span class="mono">${fmtBytes(r.bytes)}</span> of graph facts for this question`]);
  if (tier === "graph") rows.push(["Pack · model", `${esc(pk)} · ${esc(r.model || "")}`]);
  const howTitle = tier === "graph" ? "How I got this" : "Details";
  const how = m.nopack || m.streaming ? "" : `<details class="how"><summary>${howTitle}</summary><div class="how-body">
      ${tier === "graph" ? (r.path ? `<div><h2>Graph path</h2>${chainHTML(r)}</div>` : `<p class="note">No path was walked.</p>`) : ""}
      ${tier !== "graph" && tier !== "notice" && r.graph_tried && r.graph_tried.path ? `<div><h2>Path the graph tried</h2>${chainHTML(r.graph_tried)}</div>` : ""}
      <dl class="facts">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      ${tier === "lm" ? `<p class="note">Written by a small language model running on this device. It can state wrong facts confidently.</p>` : ""}</div></details>`;
  const meta = !m.nopack && r.ms && !m.streaming ? `<span class="meta">${fmtMs(r.ms)}${r.bytes ? ` · ${fmtBytes(r.bytes)}` : ""}</span>` : "";
  const sugg = m.suggest && !m.suggestDone ? `<div class="suggest">Save to About me? <q>${esc(m.suggest)}</q>
      <button class="btn" type="button" data-save>Save</button><button class="btn" type="button" data-nosave>No thanks</button></div>`
    : m.suggestDone === "saved" ? `<div class="suggest done">Saved to About me.</div>` : "";
  return `<div class="msg glm"><div class="who"><span class="logo" aria-hidden="true">G</span>GLM Pocket ${m.nopack ? "" : BADGE[tier]}</div>
    <div class="reply${reveal ? " reveal" : ""}">${head}
      <div class="tools">${m.nopack || m.streaming ? "" : `<button class="chip" type="button" data-copy aria-label="Copy answer">
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/></svg><span>Copy</span></button>`}${meta}</div>
      ${sugg}${how}</div></div>`;
}

function msgHTML(m, i, reveal) {
  if (m.role === "user") return `<div class="msg user"><div class="sr">You said:</div><div class="bubble">${esc(m.text)}</div></div>`;
  if (m.pending) return `<div class="msg glm"><div class="who"><span class="logo" aria-hidden="true">G</span>GLM Pocket</div>
    <div class="reply"><span class="thinking"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>${esc(m.status || "Walking the graph…")}</span></div></div>`;
  return replyHTML(m, reveal).replace('class="msg glm"', `class="msg glm" data-i="${i}"`);
}

// fact starters the graph answers above its confidence rule (checked in tools/glm_parity/pocket_check.mjs)
const STARTERS_GENERAL = ["Who wrote the Dark Tower series?", "What is the capital of France?"];
let revealIdx = -1;
function renderThread() {
  const c = curChat();
  $("thread").innerHTML = c ? c.msgs.map((m, i) => msgHTML(m, i, i === revealIdx && !reduced())).join("") : "";
  revealIdx = -1;
  $("hello").hidden = !!(c && c.msgs.length);
  const ex = (state.active === "general" ? STARTERS_GENERAL : model.examples(state.active).slice(0, 4).map(e => e.q)).map(q => ({ q, tag: "fact · knowledge graph" }));
  const chat = [{ q: "Give me three ideas for a quick weeknight dinner.", tag: "chat model" }, { q: "Write a haiku about autumn rain.", tag: "chat model" }];
  $("starters").innerHTML = [...ex, ...(model.lmState === "off" || model.lmState === "error" ? [] : chat)].map(e => `<button class="starter" type="button" data-q="${esc(e.q)}"><b>${esc(e.tag)}</b>${esc(e.q)}</button>`).join("");
}
$("starters").addEventListener("click", e => { const b = e.target.closest("[data-q]"); if (b && !b.disabled) ask(b.dataset.q); });

$("thread").addEventListener("click", async e => {
  const host = e.target.closest("[data-i]");
  const c = curChat(), i = host ? +host.dataset.i : -1, m = c && c.msgs[i];
  if (!m) return;
  if (e.target.closest("[data-save]")) { memory.add(m.suggest); m.suggestDone = "saved"; saveChats(); renderMemory(); renderThread(); return; }
  if (e.target.closest("[data-nosave]")) { m.suggestDone = "no"; saveChats(); renderThread(); return; }
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  const q = c.msgs[i - 1] && c.msgs[i - 1].role === "user" ? c.msgs[i - 1].text : "";
  const r = m.r;
  const text = r.tier === "notice" ? [q && `Q: ${q}`, "GLM Pocket: no answer (fact questions only without the chat model)."].filter(Boolean).join("\n")
    : r.tier === "lm" ? [q && `Q: ${q}`, `GLM Pocket (language model, may be wrong): ${r.text}`].filter(Boolean).join("\n")
    : [q && `Q: ${q}`, r.ok ? `GLM: ${r.answers.map(fmtName).join(", ")}` : "GLM: I don't know from this graph.", r.path && `Path: ${pathText(r)}`, `(${packTitle(r.pack)} pack, ${r.model})`].filter(Boolean).join("\n");
  let ok = false;
  try { await navigator.clipboard.writeText(text); ok = true; }
  catch {
    try { const t = document.createElement("textarea"); t.value = text; t.style.position = "fixed"; t.style.opacity = "0"; document.body.append(t); t.select(); ok = document.execCommand("copy"); t.remove(); } catch { ok = false; }
  }
  b.querySelector("span").textContent = ok ? "Copied" : "Copy failed";
  setTimeout(() => { b.querySelector("span").textContent = "Copy"; }, 1500);
});

function scrollEnd(smooth = true) {
  const s = $("scroll");
  requestAnimationFrame(() => s.scrollTo({ top: s.scrollHeight, behavior: smooth && !reduced() ? "smooth" : "auto" }));
}
function renderAll() { renderChats(); renderPacks(); renderMemory(); renderLM(); renderThread(); renderComposer(); }

// ------------------------------------------------------------------ ask
// earlier turns of this chat for the LM: graph answers become short assistant turns
function historyOf(c, upto) {
  const h = [];
  const ms = c.msgs.slice(0, upto);
  for (const [i, m] of ms.entries()) {
    if (m.role === "user") { if (!(ms[i + 1] && ms[i + 1].r && ms[i + 1].r.tier === "notice")) h.push({ role: "user", content: m.text }); }
    else if (m.r && m.r.tier === "notice") continue;
    else if (m.r && !m.pending) h.push({ role: "assistant", content: m.r.tier === "lm" ? m.r.text : m.r.ok && m.r.answers ? m.r.answers.join(", ") : "I don't know." });
  }
  return h.slice(-6);
}

async function ask(text) {
  const q = text.trim();
  if (!q || !state.ready || state.busy) return;
  let c = curChat();
  if (!c) {
    c = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: q.length > 60 ? q.slice(0, 57) + "…" : q, t: Date.now(), msgs: [] };
    state.chats.unshift(c); state.cur = c.id;
  } else { state.chats = [c, ...state.chats.filter(x => x !== c)]; }
  const pack = state.active;
  const hist = historyOf(c, c.msgs.length);
  c.msgs.push({ role: "user", text: q });
  const pending = { role: "glm", pending: true, pack };
  c.msgs.push(pending);
  $("q").value = ""; autosize();
  state.busy = true;
  renderAll(); scrollEnd();
  await new Promise(r => setTimeout(r, 0));   // paint the question before the encoder runs on this thread
  let msg;
  const s = model.packState(pack);
  if (!s || s === "loading") msg = { role: "glm", pack, nopack: true, r: { ok: false, answers: [], pack, model: model.id } };
  else {
    let facts = [];
    try { facts = await memory.relevant(q, model.engine && model.engine.glm && model.engine.glm.enc); } catch (e) { console.warn("[pocket] memory ranking failed", e); facts = memory.list().slice(0, 5); }
    const live = { role: "glm", pack, streaming: true, r: { tier: "lm", text: "", model: model.lmKey ? LMS[model.lmKey].id : "" } };
    let shown = false;
    const onToken = t => {
      live.r.text = t;
      if (!shown) { c.msgs[c.msgs.indexOf(pending)] = live; shown = true; if (state.cur === c.id) renderThread(); }
      const el = state.cur === c.id && $("thread").querySelector(".msg.glm:last-child .lmtext");
      if (el) el.innerHTML = prose(t);
      const sc = $("scroll"); if (sc.scrollHeight - sc.scrollTop - sc.clientHeight < 120) sc.scrollTop = sc.scrollHeight;
    };
    let r;
    const onStatus = t => { pending.status = t; if (state.cur === c.id && !shown) renderThread(); };
    try { r = await model.answer(q, { pack, history: hist, facts, onToken, onStatus }); }
    catch (e) { console.error("[pocket] answer failed", e); r = { ok: false, tier: "graph", answers: [], reason: "Something failed: " + (e.message || e), pack, model: model.id }; }
    msg = { role: "glm", pack, r };
    const sf = suggestFact(q);
    if (sf && !memory.list().some(x => x.text.toLowerCase() === sf.toLowerCase())) msg.suggest = sf;
  }
  const at = c.msgs.findIndex(m => m.pending || m.streaming);
  c.msgs[at >= 0 ? at : c.msgs.length - 1] = msg;
  state.busy = false;
  saveChats();
  if (state.cur === c.id) revealIdx = c.msgs.indexOf(msg);
  renderAll(); scrollEnd();
  if (matchMedia("(hover: hover)").matches) $("q").focus();
}

// ------------------------------------------------------------------ composer
const autosize = () => { const q = $("q"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 200) + "px"; };
$("q").addEventListener("input", () => { autosize(); renderComposer(); });
$("q").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (canAsk()) ask($("q").value); }
});
$("form").addEventListener("submit", e => { e.preventDefault(); if (canAsk()) ask($("q").value); });

// ------------------------------------------------------------------ drawer (narrow screens)
const narrow = () => matchMedia("(max-width: 860px)").matches;
function openDrawer() {
  $("app").classList.add("drawer"); $("scrim").hidden = false; $("open-side").setAttribute("aria-expanded", "true");
  document.querySelector(".main").inert = true;
  setTimeout(() => $("new").focus(), 50);
}
function closeDrawer() {
  if (!$("app").classList.contains("drawer")) return;
  $("app").classList.remove("drawer"); $("scrim").hidden = true; $("open-side").setAttribute("aria-expanded", "false");
  document.querySelector(".main").inert = false;
  if (narrow()) $("open-side").focus();
}
$("open-side").addEventListener("click", openDrawer);
$("close-side").addEventListener("click", closeDrawer);
$("scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
matchMedia("(max-width: 860px)").addEventListener("change", e => { if (!e.matches) closeDrawer(); });

// ------------------------------------------------------------------ test hook (tools/glm_parity/pocket_check.mjs)
window.glmPocket = {
  get ready() { return canAsk(); },
  ask: q => ask(q),
  pick, enableLM, memory, model, intentOf, gpu: () => state.gpu,
  chat: () => curChat(),
  engineAsk: (q, pack) => model.engine.ask(q, { pack }),
};

// ------------------------------------------------------------------ boot
async function boot() {
  state.chats = loadChats();
  const want = new URLSearchParams(location.search).get("pack");
  showLoad("Starting…", null);
  renderAll();
  const pref = getPref();
  TieredModel.webgpu().then(g => {
    state.gpu = g;
    state.lmKey = pref && LMS[pref.key] && (LMS[pref.key].engine === g.engine) ? pref.key : g.key;
    renderLM();
    // the user turned the chat model on before: load it now, from this browser's cache (no download after the first time)
    if (pref) enableLM(state.lmKey);
  });
  await model.init(p => showLoad(p.text, p.frac));
  state.active = model.catalog().some(p => p.id === want && p.status === "available") ? want : "general";
  state.ready = true;
  renderAll();
  await loadPack(state.active);
  if (matchMedia("(hover: hover)").matches) $("q").focus();
}
boot().catch(e => { console.error("[pocket] boot failed", e); showLoad(`The model did not load: ${e.message || e}`, null, true); });
