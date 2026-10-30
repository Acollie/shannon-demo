// GLM chat page: UI only. All model work goes through the adapter in ./model.js (swap the model there).
// Each question is answered on its own: there is no conversation memory yet, so earlier turns are not sent to the model.
import { GLMModel } from "./model.js";

const $ = id => document.getElementById(id);
const BASE = new URL("../../", import.meta.url).href;
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = v => (v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString("en-US")) + " ms";
const fmtBytes = b => (b < 1024 ? b + " B" : b < 1048576 ? Math.round(b / 1024) + " kB" : (b / 1048576).toFixed(1) + " MB");
const fmtName = s => (/^-?\d+(\.\d+)?$/.test(s) && Math.abs(+s) >= 10000 ? (+s).toLocaleString("en-US") : s);
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const KEY = "glm-chat-v1", MAX_CHATS = 40, SHOW = 8;

const model = new GLMModel(BASE);
const state = { ready: false, busy: false, active: "general", chats: [], cur: null, error: null };

// ------------------------------------------------------------------ storage (per-browser convenience; may be blocked)
function loadChats() { try { const v = JSON.parse(localStorage.getItem(KEY) || "[]");
  return Array.isArray(v) ? v.filter(c => c && Array.isArray(c.msgs)).map(c => ({ ...c, msgs: c.msgs.filter(m => !m.pending) })) : []; } catch { return []; } }
function saveChats() { try { localStorage.setItem(KEY, JSON.stringify(state.chats.slice(0, MAX_CHATS))); } catch { /* storage blocked or full */ } }
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
  $("top-pack").textContent = c ? c.title : `GLM · ${packTitle(state.active)} pack`;
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

// ------------------------------------------------------------------ sidebar: knowledge packs
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
  try {
    await model.loadPack(id, p => showLoad(p.text, p.frac));
    hideLoad();
  } catch (e) { console.error("[chat] pack load failed", e); showLoad(`The ${packTitle(id)} pack did not load: ${e.message || e}`, null, true); }
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
  q.placeholder = !state.ready ? "Loading the model…" : s === "loading" ? `Loading the ${packTitle(state.active)} pack…`
    : !s ? `Load the ${packTitle(state.active)} pack to ask (Knowledge packs)` : `Ask the ${packTitle(state.active)} pack…`;
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

function replyHTML(m, reveal) {
  const r = m.r, pk = packTitle(r.pack || m.pack);
  let head;
  if (m.nopack) head = `<p class="idk">The ${esc(pk)} pack is not loaded, so there is no graph to answer from.</p><p class="idk-why">Load it under Knowledge packs (menu), then ask again.</p>`;
  else if (r.ok) {
    const shown = r.answers.slice(0, SHOW).map(a => `<span class="a" data-raw="${esc(a)}">${esc(fmtName(a))}</span>`).join("");
    const more = r.answers.length > SHOW ? `<span class="more">+${r.answers.length - SHOW} more</span>` : "";
    head = `<div class="lab">Answer</div><div class="answers">${shown}${more}</div>`;
  } else head = `<p class="idk">I don't know from this graph.</p>${r.reason ? `<p class="idk-why">${esc(r.reason)}</p>` : ""}`;
  const rows = [];
  if (r.type) rows.push(["Answer type", esc(r.type) + (r.explain ? ` · ${esc(r.explain)}` : "")]);
  if (r.names && r.names.length) rows.push(["Names found", [...new Set(r.names)].map(n => esc(fmtName(n))).join(", ")]);
  if (r.score != null) rows.push(["Path score", `<span class="mono">${r.score.toFixed(2)}</span> (raw, not a probability)`]);
  if (r.ms) rows.push(["Time", `<span class="mono">${fmtMs(r.ms)}</span> on this device`]);
  if (r.bytes) rows.push(["Downloaded", `<span class="mono">${fmtBytes(r.bytes)}</span> of graph facts for this question`]);
  rows.push(["Pack · model", `${esc(pk)} · ${esc(r.model || "")}`]);
  const how = m.nopack ? "" : `<details class="how"><summary>How I got this</summary><div class="how-body">
      ${r.path ? `<div><h2>Graph path</h2>${chainHTML(r)}</div>` : `<p class="note">No path was walked.</p>`}
      <dl class="facts">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      <p class="note">This answer used only this question; earlier messages are not read.</p></div></details>`;
  const meta = !m.nopack && r.ms ? `<span class="meta">${fmtMs(r.ms)}${r.bytes ? ` · ${fmtBytes(r.bytes)}` : ""}</span>` : "";
  return `<div class="msg glm"><div class="who"><span class="logo" aria-hidden="true">G</span>GLM · ${esc(pk)}</div>
    <div class="reply${reveal ? " reveal" : ""}">${head}
      <div class="tools">${m.nopack ? "" : `<button class="chip" type="button" data-copy aria-label="Copy answer">
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/></svg><span>Copy</span></button>`}${meta}</div>
      ${how}</div></div>`;
}

function msgHTML(m, i, reveal) {
  if (m.role === "user") return `<div class="msg user"><div class="sr">You said:</div><div class="bubble">${esc(m.text)}</div></div>`;
  if (m.pending) return `<div class="msg glm"><div class="who"><span class="logo" aria-hidden="true">G</span>GLM · ${esc(packTitle(m.pack))}</div>
    <div class="reply"><span class="thinking"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>Walking the graph…</span></div></div>`;
  return replyHTML(m, reveal).replace('class="msg glm"', `class="msg glm" data-i="${i}"`);
}

let revealIdx = -1;
function renderThread() {
  const c = curChat();
  $("thread").innerHTML = c ? c.msgs.map((m, i) => msgHTML(m, i, i === revealIdx && !reduced())).join("") : "";
  revealIdx = -1;
  $("hello").hidden = !!(c && c.msgs.length);
  $("hello-pack").textContent = `${packTitle(state.active).toLowerCase()} graph`;
  $("starters").innerHTML = model.examples(state.active).map(e => `<button class="starter" type="button" data-q="${esc(e.q)}"><b>${esc(e.tag)}</b>${esc(e.q)}</button>`).join("");
}
$("starters").addEventListener("click", e => { const b = e.target.closest("[data-q]"); if (b && !b.disabled) ask(b.dataset.q); });

$("thread").addEventListener("click", async e => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  const c = curChat(), i = +b.closest("[data-i]").dataset.i, m = c && c.msgs[i];
  if (!m) return;
  const q = c.msgs[i - 1] && c.msgs[i - 1].role === "user" ? c.msgs[i - 1].text : "";
  const r = m.r;
  const text = [q && `Q: ${q}`, r.ok ? `GLM: ${r.answers.map(fmtName).join(", ")}` : "GLM: I don't know from this graph.", r.path && `Path: ${pathText(r)}`, `(${packTitle(r.pack)} pack, ${r.model})`].filter(Boolean).join("\n");
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

function renderAll() { renderChats(); renderPacks(); renderThread(); renderComposer(); }

// ------------------------------------------------------------------ ask
async function ask(text) {
  const q = text.trim();
  if (!q || !state.ready || state.busy) return;
  let c = curChat();
  if (!c) {
    c = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: q.length > 60 ? q.slice(0, 57) + "…" : q, t: Date.now(), msgs: [] };
    state.chats.unshift(c); state.cur = c.id;
  } else { state.chats = [c, ...state.chats.filter(x => x !== c)]; }
  const pack = state.active;
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
  else msg = { role: "glm", pack, r: await model.answer(q, { pack }) };
  c.msgs[c.msgs.indexOf(pending)] = msg;
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

// ------------------------------------------------------------------ test hook (tools/glm_parity/chat_parity.mjs)
window.glmChat = {
  get ready() { return canAsk(); },
  ask: q => ask(q),
  pick,
  engineAsk: (q, pack) => model.engine.ask(q, { pack }),    // the engine itself, for parity checks
  model,
};

// ------------------------------------------------------------------ boot
async function boot() {
  state.chats = loadChats();
  const want = new URLSearchParams(location.search).get("pack");
  showLoad("Starting…", null);
  renderAll();
  await model.init(p => showLoad(p.text, p.frac));
  state.active = model.catalog().some(p => p.id === want && p.status === "available") ? want : "general";
  state.ready = true;
  renderAll();
  await loadPack(state.active);
  if (matchMedia("(hover: hover)").matches) $("q").focus();
}
boot().catch(e => { console.error("[chat] boot failed", e); showLoad(`The model did not load: ${e.message || e}`, null, true); });
