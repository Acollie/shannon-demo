// GLM v0 page: UI. Model work is in engine.js (main thread; ~15 ms per question).
import { Engine } from "./engine.js";
const $ = id => document.getElementById(id);
const BASE = new URL("./", location.href).href;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ms = v => (v < 10 ? v.toFixed(1) : Math.round(v)) + " ms";
const mb = b => (b / 1048576).toFixed(1) + " MB";

const state = { ready: false, packs: new Map(), catalog: [], busy: false, rid: 0, pending: new Map(), loadT0: 0 };

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
    const extra = live && live.bytes ? ` · ${mb(live.bytes)} · built in ${ms(live.build_ms)}` : (p.bytes ? ` · ${mb(p.bytes)}` : "");
    return `<div class="pack"><div class="meta"><div class="t">${esc(p.title)} ${tag}</div><div class="d">${esc(p.detail)}${extra}</div></div>${btn}</div>`;
  }).join("");
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
  if (!link) return esc(q);
  return esc(q.slice(0, link.a)) + `<mark>${esc(q.slice(link.a, link.b))}</mark>` + esc(q.slice(link.b));
}

function renderResult(q, r, sample = false) {
  const el = $("result");
  el.toggleAttribute("data-sample", sample);
  if (!r.ok) {
    el.innerHTML = `<div class="qline">${qHTML(q, r.link)}</div><p class="err">${esc(r.reason)}</p>${r.link ? `<p class="samplenote">Linked to “${esc(r.link.name)}” in the ${esc(r.pack)} pack.</p>` : ""}`;
    return;
  }
  const t = r.timings;
  const top = r.answers.slice(0, 12).map(a => `<span class="a top">${esc(a)}</span>`).join("");
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
  state.packs.set(m.id, m); renderPacks();
  status(`Ready · ${state.encoder.encoder} encoder · ${m.title} pack loaded · ${((performance.now() - state.loadT0) / 1000).toFixed(1)} s from start`, "ok");
  $("go").disabled = false;
}

async function loadPack(id) {
  state.packs.set(id, "loading"); renderPacks();
  try { packsLoaded(await engine.load(id)); }
  catch (e) { state.packs.delete(id); renderPacks(); $("go").disabled = !state.ready; fail("The pack did not load", e); }
}

function unloadPack(id) {
  engine.unload(id); state.packs.delete(id); renderPacks();
  status("Pack unloaded. Ask again: without the pack Shannon has nothing to walk.", "ok");
}

async function ask(q) {
  q = q.trim();
  if (!q || !state.ready || state.busy) return;
  state.busy = true; $("go").disabled = true;
  status("Thinking…", "busy");
  await new Promise(r => setTimeout(r, 0));   // let the status paint before the encoder runs
  try {
    const result = await engine.ask(q);
    status(result.timings.total_ms != null ? `Answered in ${ms(result.timings.total_ms)} on this device` : "No answer", "ok");
    renderResult(q, result);
  } catch (e) { fail("Something failed", e); }
  finally { state.busy = false; $("go").disabled = !state.ready; }
}
$("f").addEventListener("submit", e => { e.preventDefault(); ask($("q").value); });
$("examples").addEventListener("click", e => {
  const b = e.target.closest("[data-q]");
  if (!b) return;
  $("q").value = b.dataset.q;
  ask(b.dataset.q);
});
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
    else for (const p of state.catalog) if (p.default && p.status === "available") await loadPack(p.id);
  } catch (e) { if (my === gen) fail("The encoder did not load", e); }
  finally { $("enc").disabled = false; }
}
window.glmDemo = { ask: q => engine.ask(q), get ready() { return state.ready && state.packs.size > 0 && [...state.packs.values()].every(v => v !== "loading"); } };

$("enc").addEventListener("change", () => { setPref("glm-encoder", $("enc").value); startEngine(); });

async function boot() {
  $("enc").value = getPref("glm-encoder", "fp16") === "q8" ? "q8" : "fp16";
  const [cat, ex, sample] = await Promise.all(["glm/packs.json", "glm/examples.json", "glm/sample.json"].map(f => fetch(f).then(r => r.json())));
  state.catalog = cat.packs;
  renderPacks();
  $("examples").innerHTML = ex.examples.map(e => `<button class="ex" type="button" data-q="${esc(e.q)}"><b>${e.hop}-step</b>${esc(e.q)}</button>`).join("");
  renderResult(sample.q, sample, true);
  startEngine();
}
boot().catch(e => fail("The page could not start", e));
