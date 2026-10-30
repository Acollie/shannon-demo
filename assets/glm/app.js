// GLM page: UI. Model work is in engine.js (main thread). Movies pack: glm.js (~15 ms per question); general pack
// (Wikidata, lazily loaded shards): wiki.js.
import { Engine } from "./engine.js";
import { ConvPack, normName } from "./glm.js";
import { ThinkGraph } from "./graph.js";
const $ = id => document.getElementById(id);
const BASE = new URL("./", location.href).href;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ms = v => (v < 10 ? v.toFixed(1) : Math.round(v)) + " ms";
const mb = b => (b / 1048576).toFixed(1) + " MB";

const state = { ready: false, packs: new Map(), catalog: [], busy: false, rid: 0, pending: new Map(), loadT0: 0, lastQ: null, lastPack: null,
  active: "movies", examples: {} };
const smooth = () => (matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

function status(text, kind = "busy", frac = null) {
  $("st").className = "pill " + kind;
  $("st-t").textContent = text;
  $("bar").hidden = frac === null;
  if (frac !== null) $("bar-i").style.width = Math.round(frac * 100) + "%";
}

function getPref(k, d) { try { return localStorage.getItem(k) || d; } catch { return d; } }
function setPref(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } }

// ------------------------------------------------------------------ packs
function renderPacks() {
  $("packs").innerHTML = state.catalog.map(p => {
    const live = state.packs.get(p.id);
    const loading = live === "loading";
    let tag, btn;
    if (p.status !== "available") { tag = `<span class="tag off">coming next</span>`; btn = `<button class="btn" disabled>Load</button>`; }
    else if (loading) { tag = `<span class="tag off">loading…</span>`; btn = `<button class="btn" disabled>Loading</button>`; }
    else if (live) {
      tag = `<span class="tag">loaded</span>`;
      btn = `<button class="btn" data-unload="${p.id}">Unload</button>`;
    } else { tag = `<span class="tag off">not loaded</span>`; btn = `<button class="btn" data-load="${p.id}" ${state.ready ? "" : "disabled"}>Load</button>`; }
    const extra = live && live.lazy ? ` · core ${mb(live.bytes)} loaded; facts download as you ask (${mb(live.total_bytes)} in all)`
      : live && live.bytes ? ` · ${mb(live.bytes)} · built in ${ms(live.build_ms)}` : (p.bytes ? ` · ${mb(p.bytes)}` : "");
    return `<div class="pack"><div class="meta"><div class="t">${esc(p.title)} ${tag}</div><div class="d">${esc(p.detail)}${extra}</div></div>${btn}</div>`;
  }).join("");
  renderPicker();
}

// ------------------------------------------------------------------ pack picker (which pack the question goes to)
function renderPicker() {
  const cat = state.catalog.filter(p => p.status === "available");
  $("picker").innerHTML = cat.map(p => `<button type="button" role="tab" class="pick" data-pick="${p.id}" aria-selected="${p.id === state.active}">${esc(p.pick || p.title)}</button>`).join("");
  const cur = state.catalog.find(p => p.id === state.active);
  $("pnote").innerHTML = cur && cur.preview ? `<b>Research preview</b> — ${esc(cur.preview)}` +
    ($("enc").value === "q8" ? ` <b>The small (int8) encoder changes about 1 in 4 answers on this pack; choose “full” above for the tested model.</b>` : "") : "";
  $("pnote").hidden = !(cur && cur.preview);
  const live = state.packs.get(state.active);
  $("go").disabled = !state.ready || !live || live === "loading" || state.busy;
}
function renderExamples() {
  const ex = state.examples[state.active] || [];
  $("examples").innerHTML = ex.map(e => `<button class="ex" type="button" data-q="${esc(e.q)}"><b>${esc(e.tag)}</b>${esc(e.q)}</button>`).join("");
  const cur = state.catalog.find(p => p.id === state.active);
  $("q").placeholder = cur && cur.placeholder ? cur.placeholder : "e.g. who directed Memento";
}
async function pick(id) {
  if (id === state.active && state.packs.get(id)) return;
  state.active = id; setPref("glm-pack", id);
  renderPicker(); renderExamples();
  if (!state.packs.get(id) && state.ready) await loadPack(id);
}


// ------------------------------------------------------------------ result rendering
function chainHTML(r) {
  const parts = [`<div class="node topic"><span class="n">topic</span><span class="s">${esc(r.topic)}</span></div>`];
  r.hops.forEach((h, i) => {
    const last = i === r.hops.length - 1;
    const more = h.n > h.sample.length ? ` +${(h.n - h.sample.length).toLocaleString()} more` : "";
    parts.push(`<span class="rel" title="${esc(h.edge)}"><span>${esc(h.label)}</span></span>`);
    parts.push(`<div class="node${last ? " end" : ""}"><span class="n">${h.n.toLocaleString()} ${h.n === 1 ? "node" : "nodes"}</span><span class="s">${esc(h.sample.join(", "))}${more}</span></div>`);
  });
  return `<div class="chain">${parts.join("")}</div>`;
}

function qHTML(q, link) {
  if (!link || link.a == null) return esc(q);
  return esc(q.slice(0, link.a)) + `<mark>${esc(q.slice(link.a, link.b))}</mark>` + esc(q.slice(link.b));
}

// ------------------------------------------------------------------ thinking graph
let graph = null;
function graphHTML(r) {
  if (!r.viz) return `<div class="think"><h2>Thinking graph</h2><p class="samplenote">The thinking graph is drawn when you ask a question on this device.</p></div>`;
  const v = r.viz;
  const stats = v.stats.map(s => `step ${s.hop} (${esc(s.label)}): ${s.shown} of ${s.n.toLocaleString()}`).join(" · ");
  const label = `Thinking graph. Topic ${r.topic}; path ${r.hops.map(h => h.label).join(", then ")}; answers ${r.answers.slice(0, 5).join(", ")}.`;
  return `<div class="think"><h2>Thinking graph</h2>
    <div class="gwrap"><canvas id="gc" role="img" aria-label="${esc(label)}"></canvas>
      ${v.hidden ? `<span class="gbadge" title="nodes reached by these paths but not drawn">+${v.hidden.toLocaleString()} more</span>` : ""}
      <button class="btn gre" id="greplay" type="button">Replay</button></div>
    <div class="legend"><span><i class="lg"></i>chosen path</span><span><i class="lg alt"></i>next-best paths</span>${r.kind === "wikidata" ? "" : `<span><i class="lg taught"></i>taught fact</span>`}<span><i class="lg seen"></i>looked at, not taken</span></div>
    <p class="gstats">Dots shown per step, of the nodes reached: ${stats}. ${v.nodes.length} dots drawn of ${v.reached.toLocaleString()} reached.</p>
    <div class="gfacts" id="gfacts" aria-live="polite">Tap a dot to see its facts.</div></div>`;
}

async function showFacts(id) {
  const p = engine && engine.pack(state.lastPack);
  if (p && p.kind === "wikidata") {
    $("gfacts").textContent = "Looking up its facts…";
    try {
      const f = await p.facts(id);
      const groups = f.groups.map(g => `<li><span class="rl">${esc(g.label)}</span> ${g.items.map(x => esc(fmtName(x.name))).join(", ")}${g.n > g.items.length ? ` <span class="mono">+${(g.n - g.items.length).toLocaleString()} more</span>` : ""}</li>`).join("");
      $("gfacts").innerHTML = `<b>${esc(f.name)}</b><ul>${groups || "<li>No facts.</li>"}</ul>`;
    } catch (e) { $("gfacts").textContent = "Could not load its facts: " + (e.message || e); }
    return;
  }
  if (!p || (id >= p.n && !p.ov) || (p.ov && id >= p.n + p.ov.names.length)) { $("gfacts").textContent = "That fact has been forgotten (the conversation pack was unloaded)."; return; }
  const f = p.facts(id);
  const groups = f.groups.map(g => `<li><span class="rl">${esc(g.label)}</span> ${g.items.map(x => x.taught ? `<span class="taughtx">${esc(x.name)} (taught)</span>` : esc(x.name)).join(", ")}${g.n > g.items.length ? ` <span class="mono">+${(g.n - g.items.length).toLocaleString()} more</span>` : ""}</li>`).join("");
  $("gfacts").innerHTML = `<b>${esc(f.name)}</b>${f.isNew ? ` <span class="tag t">new, taught</span>` : ""}<ul>${groups || "<li>No facts.</li>"}</ul>`;
}

const fmtName = s => (/^-?\d+(\.\d+)?$/.test(s) && Math.abs(+s) >= 10000 ? (+s).toLocaleString("en-US") : s);
const kb = b => (b < 1048576 ? Math.round(b / 1024) + " kB" : mb(b));

function renderWiki(q, r, el) {
  const t = r.timings;
  const top = r.answers.map(a => `<span class="a top">${esc(fmtName(a))}</span>`).join("");
  const sentence = [r.topic, ...r.hops.flatMap((h, i) => [h.label, i === r.hops.length - 1 ? (h.sample.slice(0, 3).map(fmtName).join(", ") + (h.n > 3 ? ", …" : "")) : (h.n === 1 ? fmtName(h.sample[0]) : `(${h.n.toLocaleString()} things)`)])].join(" → ");
  const alts = r.alternatives.map(a => `<tr><td class="p">${esc(a.from)} → ${esc(a.labels.join(" → "))}</td><td>${a.two ? `<span class="mono">${a.n.toLocaleString()} ${a.n === 1 ? "path" : "paths"}</span>` : esc(a.answers.map(fmtName).join(", ")) + (a.n > 3 ? ` <span class="mono">(${a.n.toLocaleString()})</span>` : "")}</td><td class="num">${a.score.toFixed(2)}</td></tr>`).join("");
  const others = r.op === "set" && r.others.length ? `<div class="rest">Also reached: ${r.others.slice(0, 8).map(([n, c]) => `${esc(fmtName(n))} <span class="mono">(${c})</span>`).join(", ")}${r.n_frontier > 9 ? ", …" : ""}</div>` : "";
  const OPN = { set: "pick", count: "count", yesno: "yes / no", argmax: "largest", argmin: "smallest", compare: "compare" };
  const opline = `<p class="opline"><span class="optag">${esc(OPN[r.op] || r.op)}</span> ${esc(r.explain_op)}</p>`;
  const linked = r.topics_named && r.topics_named.length ? `<p class="samplenote">Names found in the question: ${r.topics_named.map(esc).join(", ")}.</p>` : "";
  el.innerHTML = `
    <div class="qline">${qHTML(q, r.link)} <span class="mono">· general pack</span></div>
    <div><h2>Answer</h2><div class="answers">${top || `<span class="a">no answer</span>`}</div>${opline}${others}</div>
    ${r.hops.length ? graphHTML(r) : ""}
    <div><h2>Reasoning path · score ${r.score.toFixed(2)}</h2>${r.hops.length ? chainHTML({ ...r, hops: r.hops.map(h => ({ ...h, sample: h.sample.map(fmtName), edge: h.label })) }) : "<p>No path from the linked names leads anywhere.</p>"}<p class="sentence">${esc(sentence)}</p>${linked}</div>
    ${alts ? `<div><h2>Next best paths (of ${r.n_full.toLocaleString()} that exist from the names found)</h2>
      <div class="tw"><table><thead><tr><th>path</th><th>would answer</th><th style="text-align:right">score</th></tr></thead><tbody>${alts}</tbody></table></div></div>` : ""}
    <div><h2>Timing on this device</h2><div class="timing">
      <div><b>${ms(t.link_ms)}</b><span>find the names (incl. download)</span></div>
      <div><b>${ms(t.encoder_ms)}</b><span>MiniLM encoder</span></div>
      <div><b>${ms(t.fetch_ms + t.paths_ms)}</b><span>list paths (incl. download)</span></div>
      <div><b>${ms(t.score_ms + t.tagger_ms)}</b><span>score</span></div>
      <div><b>${ms(t.walk_ms)}</b><span>walk + answer</span></div>
      <div class="total"><b>${ms(t.total_ms)}</b><span>total · downloaded ${kb(t.download_bytes)} (${t.download_files} files)</span></div>
    </div></div>`;
  if (r.viz) {
    graph = new ThinkGraph($("gc"), { onSelect: n => showFacts(n.id) });
    graph.show(r.viz);
    $("greplay").addEventListener("click", () => graph && graph.replay());
  }
}

function renderResult(q, r, sample = false) {
  const el = $("result");
  if (graph) { graph.destroy(); graph = null; }
  el.toggleAttribute("data-sample", sample);
  if (r.kind === "wikidata" && r.ok) return renderWiki(q, r, el);
  if (!r.ok) {
    el.innerHTML = `<div class="qline">${qHTML(q, r.link)}</div><p class="err">${esc(r.reason || "No answer.")}</p>${r.link ? `<p class="samplenote">Linked to “${esc(r.link.name)}” in the ${esc(r.pack)} pack.</p>` : ""}`;
    return;
  }
  const t = r.timings;
  const taught = new Set(r.viz ? r.viz.taught_answers : []);
  const top = r.answers.slice(0, 12).map(a => taught.has(a) ? `<span class="a top taughtx" title="reached through a taught fact">${esc(a)}<small>taught</small></span>` : `<span class="a top">${esc(a)}</span>`).join("");
  const moreTop = r.answers.length > 12 ? `<span class="rest">+${r.answers.length - 12} more with the same support</span>` : "";
  const rest = r.others.length ? `<div class="rest">Also reached: ${r.others.slice(0, 8).map(([n, c]) => `${esc(n)} <span class="mono">(${c})</span>`).join(", ")}${r.n_frontier - r.answers.length > 8 ? ", …" : ""}</div>` : "";
  const sentence = [r.topic, ...r.hops.flatMap((h, i) => [h.edge, i === r.hops.length - 1
    ? r.answers.slice(0, 3).join(", ") + (r.answers.length > 3 ? ", …" : "")
    : (h.n === 1 ? h.sample[0] : `(${h.n.toLocaleString()} nodes)`)])].join(" → ");
  const alts = r.alternatives.map(a => `<tr><td class="p">${esc(a.path.join(" → "))}</td><td>${esc(a.answers.slice(0, 3).join(", "))}${a.n > 3 ? ` <span class="mono">(${a.n.toLocaleString()})</span>` : ""}</td><td class="num">${a.score.toFixed(2)}</td></tr>`).join("");
  el.innerHTML = `
    ${sample ? `<p class="samplenote">Example answer, computed ahead of time. Ask your own question once the encoder is ready.</p>` : ""}
    <div class="qline">${qHTML(q, r.link)} <span class="mono">· ${esc(r.pack)} pack</span></div>
    <div><h2>Answer</h2><div class="answers">${top}${moreTop}</div>${rest}</div>
    ${graphHTML(r)}
    <div><h2>Reasoning path · score ${r.score.toFixed(2)}${r.walked_rank ? ` · rank ${r.walked_rank + 1} (higher paths reached nothing)` : ""}</h2>${chainHTML(r)}<p class="sentence">${esc(sentence)}</p></div>
    <div><h2>Next best paths (of ${r.n_cands.toLocaleString()} that exist from ${esc(r.topic)})</h2>
      <div class="tw"><table><thead><tr><th>path</th><th>would answer</th><th style="text-align:right">score</th></tr></thead><tbody>${alts}</tbody></table></div></div>
    <div><h2>${sample ? "Timing (laptop, Node)" : "Timing on this device"}</h2><div class="timing">
      <div><b>${ms(t.link_ms)}</b><span>find the entity</span></div>
      <div><b>${ms(t.encoder_ms)}</b><span>MiniLM encoder</span></div>
      <div><b>${ms(t.tagger_ms)}</b><span>tagger + operator</span></div>
      <div><b>${ms(t.paths_ms)}</b><span>list paths</span></div>
      <div><b>${ms(t.score_ms + t.walk_ms)}</b><span>score + walk</span></div>
      <div class="total"><b>${ms(t.total_ms)}</b><span>total</span></div>
    </div></div>`;
  if (r.viz) {
    graph = new ThinkGraph($("gc"), { onSelect: n => showFacts(n.id) });
    graph.show(r.viz);
    $("greplay").addEventListener("click", () => graph && graph.replay());
  }
}

// ------------------------------------------------------------------ engine (main thread)
const STEP_TEXT = {
  "runtime script": "Loading the runtime script…", "runtime wasm": "Loading the runtime (WebAssembly)…",
  "encoder manifest": "Loading the encoder manifest…", "vocabulary": "Loading the vocabulary…",
  "encoder download": "Loading the encoder…", "encoder session": "Starting the encoder…", "warm-up": "Warming up the encoder…",
};
function onStatus({ step, done, total }) {
  const t = STEP_TEXT[step] || (step.startsWith("pack ") ? `Loading the ${step.slice(5).split(":")[0]} pack… (${step.split(": ")[1]})` : step + "…");
  if (done != null) status(`${t} ${total ? `${mb(done)} of ${mb(total)}` : mb(done)}`, "busy", total ? done / total : null);
  else status(t, "busy", null);
}
function fail(what, e) {
  console.error(`[glm] ${what}`, e);
  status(`${what}: ${e && e.message || e}`, "bad");
}

function packsLoaded(m) {
  state.packs.set(m.id, m); renderPacks(); teachReady();
  status(`Ready · ${state.encoder.encoder} encoder · ${m.title} pack loaded · ${((performance.now() - state.loadT0) / 1000).toFixed(1)} s from start`, "ok");
  renderPicker();
}

async function loadPack(id) {
  state.packs.set(id, "loading"); renderPacks();
  try { packsLoaded(await engine.load(id)); }
  catch (e) { state.packs.delete(id); renderPacks(); fail("The pack did not load", e); }
}

function unloadPack(id) {
  if (teach.conv && teach.conv.base === engine.pack(id)) { teach.conv.detach(); teach.conv = null; teach.facts = []; teach.h = {}; renderTeach("The movies pack was unloaded, so the conversation pack was dropped too."); }
  engine.unload(id); state.packs.delete(id); renderPacks(); teachReady();
  status("Pack unloaded. Ask again: without the pack Shannon has nothing to walk.", "ok");
  renderPicker();
}

async function ask(q) {
  q = q.trim();
  if (!q || !state.ready || state.busy) return;
  const live = state.packs.get(state.active);
  if (!live || live === "loading") { status("Load the pack first (Knowledge packs, above).", "bad"); return; }
  state.busy = true; $("go").disabled = true;
  status(state.active === "movies" ? "Thinking…" : "Thinking… (fetching the facts it needs)", "busy");
  await new Promise(r => setTimeout(r, 0));   // let the status paint before the encoder runs
  try {
    const result = await engine.ask(q, { viz: true, pack: state.active });
    state.lastQ = q; state.lastPack = result.pack || state.lastPack;
    status(result.timings.total_ms != null ? `Answered in ${ms(result.timings.total_ms)} on this device${result.timings.download_bytes ? ` (downloaded ${kb(result.timings.download_bytes)})` : ""}` : "No answer", "ok");
    renderResult(q, result);
  } catch (e) { fail("Something failed", e); }
  finally { state.busy = false; renderPicker(); }
}
$("f").addEventListener("submit", e => { e.preventDefault(); ask($("q").value); });
$("examples").addEventListener("click", e => {
  const b = e.target.closest("[data-q]");
  if (!b) return;
  $("q").value = b.dataset.q;
  ask(b.dataset.q);
});
$("picker").addEventListener("click", e => { const b = e.target.closest("[data-pick]"); if (b) pick(b.dataset.pick); });
$("packs").addEventListener("click", e => {
  const l = e.target.closest("[data-load]"), u = e.target.closest("[data-unload]");
  if (l) loadPack(l.dataset.load);
  if (u) unloadPack(u.dataset.unload);
});

let engine = null, gen = 0;
async function startEngine() {
  const my = ++gen;
  if (!engine) engine = new Engine(BASE, m => { if (my === gen) onStatus(m); });
  else engine.onStatus = m => { if (my === gen) onStatus(m); };
  state.ready = false; renderPacks();
  $("go").disabled = true; $("enc").disabled = true;
  state.loadT0 = performance.now();
  try {
    const m = await engine.init($("enc").value);
    if (my !== gen) return;
    state.ready = true; state.encoder = m;
    status(`Encoder ready (${m.encoder}, ${mb(m.bytes)}) in ${(m.load_ms / 1000).toFixed(1)} s`, "ok");
    renderPacks();
    const live = [...state.packs.entries()].filter(([, v]) => v && v !== "loading");
    if (live.length) packsLoaded(live[0][1]);
    else for (const p of state.catalog) if ((p.default || p.id === state.active) && p.status === "available") await loadPack(p.id);
  } catch (e) { if (my === gen) fail("The encoder did not load", e); }
  finally { $("enc").disabled = false; }
}
// ------------------------------------------------------------------ teach & forget (conversation pack)
const teach = { conv: null, facts: [], h: {}, busy: false };
const SCRIPTS = [
  { label: "Teach a new film, then ask", facts: [["Christopher Nolan", "directed_by", 1, "Example Film"]], q: "what films did Christopher Nolan direct",
    text: "Teach: Christopher Nolan → director of → Example Film. Ask: what films did Christopher Nolan direct?" },
  { label: "Teach two facts, ask a 2-step question", facts: [["Paper Moonlight", "directed_by", 0, "Christopher Nolan"], ["Paper Moonlight", "in_language", 0, "French"]],
    q: "what other films did the director of Paper Moonlight direct",
    text: "Teach: Paper Moonlight → directed by → Christopher Nolan, and Paper Moonlight → in language → French. Ask: what other films did the director of Paper Moonlight direct?" },
];
const basePack = () => { const v = state.packs.get("movies"); return v && v !== "loading" && engine ? engine.pack("movies") : null; };   // teaching works on the movies pack

async function sha(p) {
  if (!globalThis.crypto || !crypto.subtle) return "(Web Crypto is not available on this page)";
  const d = await crypto.subtle.digest("SHA-256", p.digestBytes());
  return Array.from(new Uint8Array(d), b => b.toString(16).padStart(2, "0")).join("");
}

function teachReady() {
  const p = basePack(), on = !!p && state.ready;
  $("t-add").disabled = !on;
  for (const b of document.querySelectorAll(".script")) b.disabled = !on;
  if (p && $("t-r").dataset.pack !== p.id) {
    $("t-r").innerHTML = p.edgeLabels.map((l, e) => `<option value="${e}">${esc(l)}</option>`).join("");
    $("t-r").dataset.pack = p.id;
  }
  if (!p) $("t-r").innerHTML = "", delete $("t-r").dataset.pack;
}

function renderTeach(msg) {
  if (msg != null) $("t-msg").textContent = msg;
  const p = teach.conv && teach.conv.base, live = !!teach.conv;
  const rel = f => p ? p.edgeLabels[2 * f.r] : f.r;
  $("t-facts").innerHTML = teach.facts.map((f, i) => `<li class="${live ? "" : "gone"}"><span>${esc(f.h)} <span class="rl mono">${esc(f.rel || rel(f))}</span> ${esc(f.t)}${f.hNew || f.tNew ? ` <span class="tag t">new name</span>` : ""}</span>${live ? `<button class="x" type="button" data-rm="${i}">remove</button>` : ""}</li>`).join("");
  $("t-forget").disabled = !live;
  const h = teach.h, row = (k, t) => h[k] ? `<div><span>${t}</span><code>${esc(h[k])}</code></div>` : "";
  let verdict = "";
  if (h.before && h.load) verdict = h.before === h.load ? `<p class="ok">Same before and while loaded: teaching did not touch the base pack.</p>` : `<p class="bad">The base pack changed while loaded.</p>`;
  if (h.before && h.unload) verdict = h.before === h.unload && h.load === h.before ? `<p class="ok">All three match: the base pack is exactly as it was. Only the conversation pack came and went.</p>` : `<p class="bad">The hashes differ.</p>`;
  $("t-hash").innerHTML = h.before ? `<p>SHA-256 of the ${esc(h.pack)} pack's graph in memory (every name and every link):</p>${row("before", "before loading the conversation pack")}${row("load", "with the conversation pack loaded")}${row("unload", "after forgetting")}${verdict}` : "";
}

async function addFact(s, e, o) {
  const p = basePack();
  if (!p) return { ok: false, reason: "Load the movies pack first." };
  if (!teach.conv || teach.conv.base !== p) {
    teach.conv = new ConvPack(p); teach.facts = [];
    teach.h = { pack: p.id, before: await sha(p) };
  }
  const r = teach.conv.add(s, e, o);
  if (!r.ok) return r;
  teach.conv.attach();
  teach.facts = teach.conv.facts.map(f => ({ ...f, rel: p.edgeLabels[2 * f.r] }));
  teach.h.load = await sha(p); delete teach.h.unload;
  return r;
}

async function forget() {
  if (!teach.conv) return;
  const p = teach.conv.base;
  teach.conv.detach(); teach.conv = null;
  teach.h.unload = await sha(p);
  renderTeach("Forgotten. The conversation pack is unloaded; its facts are gone from Shannon.");
  if (state.lastQ) { $("q").value = state.lastQ; await ask(state.lastQ); }
}

function hint(inp, out) {
  const p = basePack(), v = inp.value.trim();
  if (!p || !v) { out.textContent = ""; out.className = "hint"; return; }
  const id = new ConvPack(p).existing(v);
  if (id != null) { out.textContent = `in the ${p.id} pack · ${p.dg(id).toLocaleString()} facts`; out.className = "hint"; }
  else if (p.ov && p.ov.index.has(normName(v))) { out.textContent = "taught earlier in this conversation"; out.className = "hint new"; }
  else { out.textContent = "new name: becomes a new entity in the conversation pack"; out.className = "hint new"; }
}
let sugT = 0;
function suggest(inp, dl, out) {
  clearTimeout(sugT);
  sugT = setTimeout(() => {
    const p = basePack();
    hint(inp, out);
    if (!p || inp.value.trim().length < 2) { dl.innerHTML = ""; return; }
    dl.innerHTML = p.suggest(inp.value, 8).map(x => `<option value="${esc(x.name)}"></option>`).join("");
  }, 60);
}
$("t-s").addEventListener("input", () => suggest($("t-s"), $("dl-s"), $("h-s")));
$("t-o").addEventListener("input", () => suggest($("t-o"), $("dl-o"), $("h-o")));
$("tf").addEventListener("submit", async e => {
  e.preventDefault();
  const s = $("t-s").value, o = $("t-o").value, rel = +$("t-r").value;
  const r = await addFact(s, rel, o);
  if (r.ok) { $("t-s").value = ""; $("t-o").value = ""; $("h-s").textContent = ""; $("h-o").textContent = ""; }
  renderTeach(r.ok ? `Taught: ${s.trim()} → ${basePack().edgeLabels[rel]} → ${o.trim()}. Ask a question that uses it.` : r.reason);
});
$("t-facts").addEventListener("click", async e => {
  const b = e.target.closest("[data-rm]");
  if (!b || !teach.conv) return;
  teach.conv.remove(+b.dataset.rm); teach.conv.attach(); teach.facts = teach.conv.facts.map(f => ({ ...f, rel: teach.conv.base.edgeLabels[2 * f.r] }));
  teach.h.load = await sha(teach.conv.base);
  renderTeach("Removed that fact.");
});
$("t-forget").addEventListener("click", () => forget());

async function runScript(sc) {
  const p = basePack();
  if (!p || teach.busy) return;
  teach.busy = true;
  if (state.active !== "movies") await pick("movies");
  try {
    const msgs = [];
    for (const [s, rid, inv, o] of sc.facts) {
      const e = 2 * p.m.relations.findIndex(r => r.id === rid) + inv;
      $("t-s").value = s; $("t-r").value = e; $("t-o").value = o;
      const r = await addFact(s, e, o);
      msgs.push(r.ok ? `Taught: ${s} → ${p.edgeLabels[e]} → ${o}.` : `${s} → ${p.edgeLabels[e]} → ${o}: ${r.reason}`);
    }
    $("t-s").value = ""; $("t-o").value = "";
    renderTeach(msgs.join(" ") + ` Asking: “${sc.q}”.`);
    $("q").value = sc.q;
    await ask(sc.q);
    $("result").scrollIntoView({ behavior: smooth(), block: "start" });
  } finally { teach.busy = false; }
}
$("scripts").innerHTML = SCRIPTS.map((sc, i) => `<button class="script" type="button" data-sc="${i}" disabled><b>Example ${i + 1}: ${esc(sc.label)}</b>${esc(sc.text)}</button>`).join("");
$("scripts").addEventListener("click", e => { const b = e.target.closest("[data-sc]"); if (b && !b.disabled) runScript(SCRIPTS[+b.dataset.sc]); });

window.glmDemo = { ask: (q, pack) => engine.ask(q, pack ? { pack } : {}), pick, get active() { return state.active; }, packInfo: id => state.packs.get(id), graph: () => graph, teach: (s, e, o) => addFact(s, e, o).then(r => (renderTeach(""), r)), forget, hashes: () => ({ ...teach.h }), get ready() { return state.ready && state.packs.size > 0 && [...state.packs.values()].every(v => v !== "loading"); } };

$("enc").addEventListener("change", () => { setPref("glm-encoder", $("enc").value); renderPicker(); startEngine(); });

async function boot() {
  $("enc").value = getPref("glm-encoder", "fp16") === "q8" ? "q8" : "fp16";
  const [cat, ex, exg, sample] = await Promise.all(["glm/packs.json", "glm/examples.json", "glm/examples_general.json", "glm/sample.json"].map(f => fetch(f).then(r => r.json())));
  state.catalog = cat.packs;
  state.examples = { movies: ex.examples.map(e => ({ q: e.q, tag: `${e.hop}-step` })), general: exg.examples.map(e => ({ q: e.q, tag: e.tag })) };
  const want = getPref("glm-pack", "movies");
  state.active = state.catalog.some(p => p.id === want && p.status === "available") ? want : "movies";
  renderPacks(); renderExamples();
  renderResult(sample.q, sample, true);
  startEngine();
}
boot().catch(e => fail("The page could not start", e));
