import { Engine } from "./engine.js";
import { ShannonQA } from "./shannon_qa.js";

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const BASE = "models/qa/";
const CACHE = "shannon-qa-v1";
const FORM_NAME = { span: "span", list: "list", number: "number", bool: "yes / no", null: "not answerable" };
const MB = b => (b / 1e6).toFixed(b < 1e7 ? 1 : 0) + " MB";

const S = { manifest: null, examples: null, sys: null, readText: null, busy: false, last: null, docId: null };

// ------------------------------------------------------------------------------------------------ files
async function getJSON(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}

function versioned(f) { return `${BASE}${f.path}?v=${f.sha256}`; }

async function cacheOpen() {
  try { return "caches" in self ? await caches.open(CACHE) : null; } catch { return null; }
}

async function allCached() {
  const c = await cacheOpen();
  if (!c) return false;
  try {
    for (const f of Object.values(S.manifest.files)) if (!(await c.match(versioned(f)))) return false;
    return true;
  } catch { return false; }
}

// fetch with progress; stored in Cache Storage so the next visit loads from disk
async function fetchBytes(f, onBytes) {
  const url = versioned(f);
  const c = await cacheOpen();
  if (c) {
    try {
      const hit = await c.match(url);
      if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length, true); return b; }
    } catch { /* fall through to network */ }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${f.path}: HTTP ${r.status}`);
  const total = f.bytes;
  let got = 0;
  const parts = [];
  const rd = r.body.getReader();
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value); got += value.length; onBytes(Math.min(got, total), false);
  }
  const b = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { b.set(p, o); o += p.length; }
  if (c) {
    try {
      await c.put(url, new Response(b, { headers: { "content-type": "application/octet-stream" } }));
      for (const k of await c.keys()) if (k.url.includes(BASE + f.path + "?") && !k.url.endsWith(url.split("?")[1])) await c.delete(k);
    } catch { /* quota or private mode: still works, just not cached */ }
  }
  return b;
}

// ------------------------------------------------------------------------------------------------ boot
async function boot() {
  [S.manifest, S.examples] = await Promise.all([getJSON(BASE + "manifest.json"), getJSON("qa/examples.json")]);
  const total = S.manifest.download_bytes;
  $("load-btn").textContent = `Load Shannon QA (≈ ${Math.round(total / 1e6)} MB, once)`;
  $("load-files").innerHTML = Object.entries(S.manifest.files).map(([k, f]) =>
    `<li><span id="lf-${k}">${esc(f.path)}</span><span id="ls-${k}">${MB(f.bytes)}</span></li>`).join("");
  $("doc-chips").innerHTML = S.examples.docs.map(d => `<button type="button" data-doc="${d.id}" aria-pressed="false">${esc(d.title)}</button>`).join("") +
    `<button type="button" data-doc="" aria-pressed="false">Your own text</button>`;
  $("doc-chips").addEventListener("click", e => { const b = e.target.closest("button"); if (b) pickDoc(b.dataset.doc); });
  $("q-chips").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $("question").value = b.dataset.q;
    if (S.sys) ask(); else $("load-status").innerHTML = "Load the models first, then ask.";
  });
  $("doc").addEventListener("input", () => { docStats(); if (S.docId) markDoc(""); });
  $("ask-form").addEventListener("submit", e => { e.preventDefault(); ask(); });
  $("load-btn").addEventListener("click", load);
  await pickDoc(S.examples.docs[0].id);
  if (await allCached()) load();
}

function markDoc(id) {
  S.docId = id || null;
  for (const b of $("doc-chips").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.doc === (id || "")));
  const d = S.examples.docs.find(x => x.id === id);
  $("doc-note").textContent = d ? d.source + (d.seen ? " This document was used in development." : "") : "Paste any text. Blank lines separate paragraphs.";
  $("q-chips").innerHTML = d ? d.questions.map(q => `<button type="button" data-q="${esc(q.q)}">${esc(q.q)}</button>`).join("") : "";
}

async function pickDoc(id) {
  if (!id) { markDoc(""); $("doc").value = ""; $("doc").focus(); docStats(); return; }
  const d = S.examples.docs.find(x => x.id === id);
  const r = await fetch(d.file);
  $("doc").value = await r.text();
  markDoc(id);
  docStats();
}

function docStats() {
  const t = $("doc").value;
  const w = (t.match(/\S+/g) || []).length;
  $("doc-stats").textContent = w ? `${w.toLocaleString()} words.` : "";
}

// ------------------------------------------------------------------------------------------------ models
async function load() {
  if (S.sys || S.loading) return;
  S.loading = true;
  $("load-btn").disabled = true;
  const st = $("load-status");
  try {
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const files = S.manifest.files;
    const total = S.manifest.download_bytes;
    const got = {};
    let fromCache = true;
    const tick = k => (n, cached) => {
      got[k] = n; fromCache = fromCache && cached;
      const sum = Object.values(got).reduce((a, b) => a + b, 0);
      $("load-bar").style.width = (100 * sum / total).toFixed(1) + "%";
      $("ls-" + k).textContent = n >= files[k].bytes ? (cached ? "cached" : "done") : `${MB(n)} / ${MB(files[k].bytes)}`;
      if (n >= files[k].bytes) $("ls-" + k).className = "ok";
      st.innerHTML = `<span class="spin"></span>Downloading ${MB(sum)} of ${MB(total)}…`;
    };
    const t0 = performance.now();
    const small = {};
    for (const k of ["vocab", "yesno_vocab", "gate"]) {
      const b = await fetchBytes(files[k], tick(k));
      small[k] = new TextDecoder().decode(b);
    }
    const eng = new Engine(ort, {
      embed: files.embed, reader: files.reader, yesno: files.yesno,
      vocab: small.vocab, unigram: JSON.parse(small.yesno_vocab),
    });
    await eng.init(async (f, k) => {
      const b = await fetchBytes(f, tick(k));
      st.innerHTML = `<span class="spin"></span>Starting the ${k === "embed" ? "retrieval" : k === "reader" ? "reading" : "yes/no"} model…`;
      return b;
    }, { executionProviders: ["wasm"] });
    S.sys = new ShannonQA(eng, JSON.parse(small.gate));
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    st.textContent = `Ready. ${MB(total)} ${fromCache ? "loaded from your browser's cache" : "downloaded and cached"} in ${secs} s.`;
    $("load-btn").textContent = "Loaded";
    $("ask-btn").disabled = false;
    if (!$("question").value) $("question").placeholder = "Ask something, or tap an example question";
  } catch (e) {
    console.error(e);
    st.innerHTML = `<span class="err">Could not load the models: ${esc(e.message || e)}</span>`;
    $("load-btn").disabled = false;
  } finally { S.loading = false; }
}

// ------------------------------------------------------------------------------------------------ ask
async function ask() {
  const q = $("question").value.trim();
  const text = $("doc").value;
  if (!S.sys || S.busy || !q) return;
  if (!/\S/.test(text)) { showMsg("Paste a document first."); return; }
  S.busy = true;
  $("ask-btn").disabled = true;
  try {
    if (S.readText !== text) {
      const t0 = performance.now();
      showMsg(`<span class="spin"></span>Reading the document…`);
      await S.sys.read(text, (i, n) => showMsg(`<span class="spin"></span>Reading the document: ${i} of ${n} passages embedded…`));
      S.readText = text;
      S.readMs = performance.now() - t0;
    }
    showMsg(`<span class="spin"></span>Answering…`);
    const r = await S.sys.ask(q);
    S.last = r;
    render(r);
  } catch (e) {
    console.error(e);
    showMsg(`<span class="err">Something went wrong: ${esc(e.message || e)}</span>`);
  } finally {
    S.busy = false;
    $("ask-btn").disabled = false;
  }
}

function showMsg(html) {
  const el = $("result");
  el.classList.remove("hidden");
  el.innerHTML = `<p class="small">${html}</p>`;
}

// ------------------------------------------------------------------------------------------------ render
const PLAIN = {
  bias: "prior for this form", lpf: "question looks like this form", qa: "reader score", qa_pos: "reader prefers it to “no answer”",
  qa_r0: "reader's top span", len: "length", cos: "meaning similarity", ovl: "word overlap", emb_top: "last passage, top by meaning",
  ovl_top: "top by overlap", in_both: "top by both", nb: "number of items", digit: "written in digits", cos_max: "best meaning similarity",
  qa_best: "best reader score", qa_best_pos: "some span beats “no answer”", p_yes0: "P(yes), first passage", p_yes_max: "P(yes), max",
  p_yes_mean: "P(yes), mean", yes_bias: "prior for yes", p_no0: "P(no), first passage", p_no_max: "P(no), max", p_no_mean: "P(no), mean",
  no_bias: "prior for no", unknown: "share of question words not in the text", unknown_any: "some question word not in the text",
};
const plain = k => { const f = k.includes(":") ? k.split(":")[1] : k; return PLAIN[f] || f; };

function answerText(c) {
  if (c.form === "null") return "Not answerable from this text";
  if (c.form === "bool") return c.answer ? "Yes" : "No";
  if (c.form === "number") return Number(c.answer).toLocaleString("en-US");
  if (c.form === "list") return c.answer.length === 1 ? c.answer[0] : c.answer.join(" · ");
  return c.answer;
}

function answerHTML(c) {
  if (c.form === "list" && c.answer.length > 1) return `<ul>${c.answer.map(a => `<li>${esc(a)}</li>`).join("")}</ul>`;
  return esc(answerText(c));
}

function marked(par, ranges) {
  // ranges: [a, b, cls]; non-overlapping, sorted
  const rs = ranges.filter(r => r && r[1] > r[0]).sort((x, y) => x[0] - y[0]);
  let out = "", i = 0;
  for (const [a, b, cls] of rs) {
    if (a < i) continue;
    out += esc(par.slice(i, a)) + `<mark${cls ? ` class="${cls}"` : ""}>${esc(par.slice(a, b))}</mark>`;
    i = b;
  }
  return out + esc(par.slice(i));
}

function highlights(c) {
  const s = c.src;
  if (!s) return [];
  if (s.parts) return s.parts.map(([a, b]) => [a, b, ""]);
  if (s.span) return [[s.span[0], s.a, "weak"], [s.a, s.b, ""], [s.b, s.span[1], "weak"]];
  if (s.a !== undefined) return [[s.a, s.b, ""]];
  return [];
}

function ranks(r, pi) {
  const o = r.retrieval.byO.indexOf(pi), e = r.retrieval.byE.indexOf(pi);
  const p = [];
  if (o >= 0) p.push(`overlap #${o + 1}`);
  if (e >= 0) p.push(`meaning #${e + 1}`);
  return p.join(" · ");
}

function render(r) {
  const pars = S.sys.ev.pars;
  const win = r.cands[r.chosen];
  const order = r.cands.map((c, i) => i).sort((a, b) => r.cands[b].score - r.cands[a].score);
  const runner = r.cands[order[1]];
  const confW = Math.min(1, r.margin / 6);
  const pIdx = r.retrieval.picked;
  const pnum = pi => "P" + (pIdx.indexOf(pi) + 1);
  let h = `<div class="kicker">Answer</div><div class="qhead">${esc(r.question)}</div>`;
  h += `<div class="answer f-${win.form}"><div class="lbl">${win.form === "null" ? "Shannon QA refuses" : "Shannon QA answers"}</div>
    <div class="txt">${answerHTML(win)}</div>`;
  if (win.form === "number") h += `<div class="from">read from “${esc(pars[win.src.pi][1].slice(win.src.span[0], win.src.span[1]))}”</div>`;
  if (win.form === "null") h += `<div class="from">No candidate answer scored above “none”. The closest was ${runner ? `<b>${esc(answerText(runner))}</b> (${FORM_NAME[runner.form]})` : "nothing"}.</div>`;
  h += `</div>`;
  h += `<div class="meta">
    <div><small>Form</small><span class="tag f-${win.form}">${FORM_NAME[win.form]}</span></div>
    <div><small>Confidence (gate margin)</small><span class="conf">${r.margin.toFixed(2)}<i title="margin over the runner-up"><b style="width:${(100 * confW).toFixed(0)}%"></b></i></span></div>
    <div><small>Runner-up</small>${runner ? `${esc(answerText(runner)).slice(0, 80)} <span class="tag f-${runner.form}">${FORM_NAME[runner.form]}</span>` : "–"}</div>
    <div><small>Time</small>${(r.ms / 1000).toFixed(2)} s${S.readMs ? ` <span class="muted small">(+${(S.readMs / 1000).toFixed(1)} s to read the text, once)</span>` : ""}</div>
  </div>`;
  S.readMs = 0;

  // evidence
  h += `<div class="ev"><h3><span class="n">EVIDENCE</span>${win.form === "null" ? "Where it looked" : "The passage it cites"}</h3>`;
  if (win.form === "bool") {
    const seen = new Set();
    h += `<p class="small muted">Yes/no answers come from the cross-encoder's P(yes) on these passages; the gate combined them with the reader's scores.</p>`;
    r.topIdx.forEach((pi, k) => {
      if (seen.has(pi)) return; seen.add(pi);
      h += `<div class="ptext${k === 0 ? " used" : ""}"><span class="tag">${pnum(pi)}</span> <span class="tag f-bool">P(yes) ${r.py[k].toFixed(2)}</span> ${esc(pars[pi][1])}</div>`;
    });
  } else if (win.form === "null") {
    const best = order.map(i => r.cands[i]).find(c => c.src && c.src.a !== undefined || (c.src && c.src.parts));
    if (best) {
      h += `<p class="small muted">The best answer it turned down, marked in its passage (${pnum(best.src.pi)}, ${ranks(r, best.src.pi)}):</p>`;
      h += `<div class="ptext">${marked(pars[best.src.pi][1], highlights(best).map(x => [x[0], x[1], "weak"]))}</div>`;
    }
  } else {
    h += `<p class="small muted">${pnum(win.src.pi)} · ${ranks(r, win.src.pi)} · the cited span is marked.</p>`;
    h += `<div class="ptext used">${marked(pars[win.src.pi][1], highlights(win))}</div>`;
  }
  h += `</div>`;

  // graph
  h += `<div class="st"><h3><span class="n">HOW IT ANSWERED</span>The answer path</h3>
    <p class="small muted">Question → retrieved passages → candidate answers with their gate scores → the choice. The highlighted path is the winner.</p>
    <div class="legend">${["span", "list", "number", "bool", "null"].map(f => `<span class="tag f-${f}">${FORM_NAME[f]}</span>`).join("")}</div>
    <div class="graph" id="graph"></div></div>`;

  // all candidates
  h += `<details class="more"><summary>All ${r.cands.length} candidates and what the gate weighed</summary><div class="tablewrap"><table class="cands">
    <thead><tr><th>#</th><th>form</th><th>answer</th><th>from</th><th class="n">score</th><th>biggest pushes</th></tr></thead><tbody>`;
  order.forEach((i, k) => {
    const c = r.cands[i];
    const top = [...c.contrib].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3);
    h += `<tr class="${i === r.chosen ? "win" : ""}"><td class="n">${k + 1}</td><td><span class="tag f-${c.form}">${FORM_NAME[c.form]}</span></td>
      <td class="ans">${esc(answerText(c))}</td><td>${c.src ? pnum(c.src.pi) : "–"}</td><td class="n">${c.score.toFixed(2)}</td>
      <td>${top.map(([f, v]) => `<span class="ct ${v >= 0 ? "p" : "m"}">${esc(plain(f))} ${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>`).join("")}</td></tr>`;
  });
  h += `</tbody></table></div></details>`;
  h += `<details class="more"><summary>The ${pIdx.length} retrieved passages</summary><div style="padding:0 10px 8px">` +
    pIdx.map(pi => `<div class="ptext"><span class="tag">${pnum(pi)}</span> <span class="small muted">${ranks(r, pi)} · overlap ${r.retrieval.ovl[pi].toFixed(2)} · meaning ${r.retrieval.cos[pi].toFixed(3)}</span><br>${esc(pars[pi][1])}</div>`).join("") +
    `</div></details>`;
  const el = $("result");
  el.classList.remove("hidden");
  el.innerHTML = h;
  drawGraph(r);
}

// ------------------------------------------------------------------------------------------------ answer-path graph (SVG)
function wrapText(s, maxChars, maxLines) {
  const words = String(s).split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = "";
  for (const w of words) {
    const t = cur ? cur + " " + w : w;
    if (t.length <= maxChars) { cur = t; continue; }
    if (cur) lines.push(cur);
    cur = w.length > maxChars ? w.slice(0, maxChars - 1) + "…" : w;
    if (lines.length === maxLines) break;
  }
  if (lines.length < maxLines && cur) lines.push(cur);
  if (lines.length > maxLines) lines.length = maxLines;
  const used = lines.join(" ").length;
  if (used < String(s).replace(/\s+/g, " ").trim().length && lines.length) {
    const l = lines[lines.length - 1];
    lines[lines.length - 1] = (l.length >= maxChars ? l.slice(0, maxChars - 1) : l) + "…";
  }
  return lines;
}

function drawGraph(r) {
  const box = $("graph");
  if (!box) return;
  const W = Math.max(300, box.clientWidth || 600);
  const pars = S.sys.ev.pars;
  const pIdx = r.retrieval.picked;
  const pad = 10, gap = 10, laneH = 22;
  const cw = W < 520 ? 6.2 : 6.6;     // approx px per char at 12px
  const col = (minW, n) => { const k = Math.max(1, Math.min(n, Math.floor((W - 2 * pad + gap) / (minW + gap)))); return { k, w: (W - 2 * pad - (k - 1) * gap) / k }; };
  const order = r.cands.map((c, i) => i).sort((a, b) => r.cands[b].score - r.cands[a].score);
  // candidates to show: top 6 by score among span/list/number (distinct text+form), plus yes, no, none, and the winner
  const shown = [];
  const key = c => c.form + "|" + JSON.stringify(c.answer);
  const seen = new Set();
  for (const i of order) {
    const c = r.cands[i];
    if (c.form === "bool" || c.form === "null") continue;
    if (seen.has(key(c))) continue;
    seen.add(key(c));
    if (shown.length < (W < 520 ? 4 : 6) || i === r.chosen) shown.push(i);
  }
  for (const i of order) if (r.cands[i].form === "bool" || r.cands[i].form === "null") shown.push(i);
  shown.sort((a, b) => r.cands[b].score - r.cands[a].score);
  const smin = Math.min(...shown.map(i => r.cands[i].score)), smax = Math.max(...shown.map(i => r.cands[i].score));
  const win = r.cands[r.chosen];
  const winPi = win.src ? win.src.pi : null;
  const nodes = [], edges = [];
  let y = pad;
  const lane = t => { nodes.push({ lane: t, y: y + 12 }); y += laneH; };

  // question
  lane("QUESTION");
  const qLines = wrapText(r.question, Math.floor((W - 2 * pad - 20) / cw), 3);
  const qN = { id: "q", x: pad, y, w: W - 2 * pad, h: 14 + qLines.length * 16, lines: qLines, win: true };
  nodes.push(qN); y += qN.h + 26;

  // passages
  lane("RETRIEVED PASSAGES");
  const pc = col(150, pIdx.length);
  const pNodes = {};
  pIdx.forEach((pi, k) => {
    const c = k % pc.k, row = Math.floor(k / pc.k);
    const lines = wrapText(pars[pi][1], Math.floor((pc.w - 16) / cw), 2);
    const n = { id: "p" + pi, x: pad + c * (pc.w + gap), y: y + row * 82, w: pc.w, h: 72, title: `P${k + 1}`, sub: ranks(r, pi), lines,
      win: pi === winPi || (win.form === "bool" && pi === pIdx[0]) };
    pNodes[pi] = n; nodes.push(n);
  });
  y += Math.ceil(pIdx.length / pc.k) * 82 + 22;

  // candidates
  lane("CANDIDATE ANSWERS · GATE SCORE");
  const cc = col(140, shown.length);
  const cNodes = {};
  shown.forEach((i, k) => {
    const c = r.cands[i];
    const col_ = k % cc.k, row = Math.floor(k / cc.k);
    const lines = wrapText(answerText(c), Math.floor((cc.w - 16) / cw), 2);
    const n = { id: "c" + i, x: pad + col_ * (cc.w + gap), y: y + row * 78, w: cc.w, h: 68, form: c.form, lines,
      score: c.score, frac: smax > smin ? (c.score - smin) / (smax - smin) : 1, win: i === r.chosen, dim: i !== r.chosen };
    cNodes[i] = n; nodes.push(n);
    if (c.form === "bool") {
      for (const pi of new Set(r.topIdx)) if (pNodes[pi]) edges.push({ a: pNodes[pi], b: n, win: n.win && pi === pIdx[0] });
    } else if (c.form === "null") {
      edges.push({ a: qN, b: n, win: n.win, dash: true });
    } else if (pNodes[c.src.pi]) edges.push({ a: pNodes[c.src.pi], b: n, win: n.win });
  });
  y += Math.ceil(shown.length / cc.k) * 78 + 22;

  // choice
  lane("CHOICE");
  const fLines = wrapText(win.form === "null" ? "Refuse: not answerable from this text" : answerText(win), Math.floor((W - 2 * pad - 20) / cw), 2);
  const fN = { id: "f", x: pad, y, w: W - 2 * pad, h: 30 + fLines.length * 16, lines: fLines, form: win.form, win: true,
    sub: `margin ${r.margin.toFixed(2)} over the runner-up` };
  nodes.push(fN);
  edges.push({ a: cNodes[r.chosen], b: fN, win: true });
  y += fN.h + pad;
  if (win.form !== "bool" && win.form !== "null" && pNodes[winPi]) edges.push({ a: qN, b: pNodes[winPi], win: true });
  for (const pi of pIdx) if (pi !== winPi && !(win.form === "bool" && pi === pIdx[0])) edges.push({ a: qN, b: pNodes[pi] });
  if (win.form === "bool") edges.push({ a: qN, b: pNodes[pIdx[0]], win: true });

  const color = f => `var(--f-${f})`;
  let svg = `<svg viewBox="0 0 ${W} ${y}" width="${W}" height="${y}" role="img" aria-label="Answer path: question, retrieved passages, candidate answers, choice">`;
  const curve = (a, b) => {
    const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y, m = (y1 + y2) / 2;
    return `M${x1.toFixed(1)},${y1.toFixed(1)} C${x1.toFixed(1)},${m.toFixed(1)} ${x2.toFixed(1)},${m.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
  };
  for (const e of edges.filter(e => !e.win)) svg += `<path class="edge" d="${curve(e.a, e.b)}"${e.dash ? ' stroke-dasharray="4 4"' : ""}/>`;
  for (const e of edges.filter(e => e.win)) svg += `<path class="edge win" d="${curve(e.a, e.b)}"${e.dash ? ' stroke-dasharray="4 4"' : ""}/>`;
  for (const n of nodes) {
    if (n.lane) { svg += `<text class="lane" x="${pad}" y="${n.y}">${n.lane}</text>`; continue; }
    const cls = `node${n.win && n.id !== "q" ? " win" : ""}${n.dim && n.id[0] === "c" ? "" : ""}`;
    svg += `<g class="${cls}"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}"/>`;
    if (n.form) svg += `<rect x="${n.x}" y="${n.y}" width="4" height="${n.h}" style="fill:${color(n.form)};stroke:none"/>`;
    let ty = n.y + 16;
    if (n.title) { svg += `<text x="${n.x + 9}" y="${ty}" style="font-weight:600;font-size:12px">${n.title}</text><text class="sub" x="${n.x + 34}" y="${ty}">${esc(n.sub)}</text>`; ty += 16; }
    else if (n.id[0] === "c") { svg += `<text class="sub" x="${n.x + 10}" y="${ty}" style="fill:${color(n.form)}">${FORM_NAME[n.form]}</text><text class="sub" x="${n.x + n.w - 8}" y="${ty}" text-anchor="end">${n.score.toFixed(2)}</text>`; ty += 16; }
    for (const l of n.lines) { svg += `<text x="${n.x + 10}" y="${ty}" style="font-size:12px${n.id === "q" || n.id === "f" ? ";font-family:var(--serif);font-size:14px" : ""}">${esc(l)}</text>`; ty += 16; }
    if (n.id[0] === "c") svg += `<rect class="bar-bg" x="${n.x + 10}" y="${n.y + n.h - 10}" width="${n.w - 20}" height="4"/><rect x="${n.x + 10}" y="${n.y + n.h - 10}" width="${((n.w - 20) * n.frac).toFixed(1)}" height="4" style="fill:${color(n.form)};stroke:none"/>`;
    if (n.id === "f") svg += `<text class="sub" x="${n.x + 10}" y="${ty}">${esc(n.sub)}</text>`;
    svg += `</g>`;
  }
  svg += `</svg>`;
  box.innerHTML = svg;
}

let rz;
addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (S.last && $("graph")) drawGraph(S.last); }, 150); });

boot().catch(e => { console.error(e); $("load-status").innerHTML = `<span class="err">${esc(e.message || e)}</span>`; });
