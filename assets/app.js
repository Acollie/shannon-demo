import { WordPiece } from "./tokenizer.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const S = {
  base: "", meta: null, card: null, stats: null, dict: null, map: null, presets: [], tok: null,
  sess: null, codes: null, origCodes: null, spans: [], sel: -1, running: false, stop: false,
};

// ------------------------------------------------------------------------------------------ boot: small data files
async function getJSON(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}

async function boot() {
  const idx = await getJSON("models/models.json");
  const q = new URLSearchParams(location.search);
  const id = idx.models.includes(q.get("model")) ? q.get("model") : idx.current;
  S.base = `models/${id}/`;
  const [meta, card, stats, presets, map, vocab] = await Promise.all([
    getJSON(S.base + "export.json"), getJSON(S.base + "card.json"), getJSON(S.base + "stats.json"),
    getJSON(S.base + "presets.json"), getJSON(S.base + "map.json"), fetch(S.base + "vocab.txt").then((r) => r.text()),
  ]);
  Object.assign(S, { meta, card, stats, presets, map });
  S.tok = new WordPiece(vocab);
  const mb = Object.values(meta.export_check.sizes_bytes).reduce((a, b) => a + b, 0) / 1e6;
  S.totalBytes = Object.values(meta.export_check.sizes_bytes).reduce((a, b) => a + b, 0);
  $("model-line").textContent = `Model: ${card.name} (${card.data}). Download ≈ ${mb.toFixed(0)} MB, once; ${meta.export_check.precision}.`;
  const others = idx.models.filter((m) => m !== id);
  if (others.length) {
    $("model-line").insertAdjacentHTML("beforeend", ` Also: ${others.map((m) => `<a href="?model=${encodeURIComponent(m)}">${esc(m)}</a>`).join(", ")}.`);
  }
  $("load-btn").textContent = `Load Shannon (≈ ${mb.toFixed(0)} MB)`;
  presets.forEach((p, k) => {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = (p.title || p.text.split(/\s+/).slice(0, 9).join(" ")) + "…";
    $("preset").appendChild(o);
  });
  renderStats();
  initMap();
  updateTokInfo();
  getJSON(S.base + "dictionary.json").then((d) => { S.dict = d; initDict(); drawMap(); }).catch((e) => console.error(e));
  if (q.get("autoload") === "1") loadModel();
}

// ------------------------------------------------------------------------------------------ model loading
async function fetchWithProgress(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (!r.body) { const b = new Uint8Array(await r.arrayBuffer()); onBytes(b.length); return b; }
  const reader = r.body.getReader();
  const parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function loadModel() {
  if (S.sess || S.loading) return;
  S.loading = true;
  $("load-btn").disabled = true;
  try {
    if (!window.ort) throw new Error("onnxruntime-web did not load from the CDN");
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    let got = 0;
    const tick = (b) => {
      got += b;
      $("load-bar").style.width = `${(100 * got / S.totalBytes).toFixed(1)}%`;
      $("load-status").textContent = `downloading ${(got / 1e6).toFixed(1)} / ${(S.totalBytes / 1e6).toFixed(0)} MB`;
    };
    const names = ["encoder", "ideas", "decoder"];
    const bufs = await Promise.all(names.map((n) => fetchWithProgress(S.base + n + ".onnx", tick)));
    $("load-status").textContent = "starting the runtime…";
    const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const sess = {};
    for (let i = 0; i < 3; i++) sess[names[i]] = await ort.InferenceSession.create(bufs[i], opt);
    S.sess = sess;
    $("load-status").textContent = "ready: running on this device";
    $("load-row").classList.add("loaded");
    ["encode-btn"].forEach((b) => ($(b).disabled = false));
  } catch (e) {
    console.error(e);
    $("load-status").innerHTML = `<span class="err">Could not load: ${esc(e.message || e)}</span>`;
    $("load-btn").disabled = false;
  } finally {
    S.loading = false;
  }
}

// ------------------------------------------------------------------------------------------ text -> ideas
function prepIds() {
  const R = S.meta.arch.R, lin = S.meta.arch.lin;
  const ids = S.tok.encode($("input").value);
  const n = Math.min(lin, Math.floor(ids.length / R) * R);
  return { all: ids.length, ids: ids.slice(ids.length - n) };
}

function updateTokInfo() {
  if (!S.tok) return;
  const { all, ids } = prepIds();
  const R = S.meta.arch.R, M = S.meta.arch.n_ideas;
  let s = `${all} word-pieces → ${ids.length / R} ideas.`;
  if (all > S.meta.arch.lin) s += ` Only the last ${S.meta.arch.lin} are used.`;
  if (ids.length / R < M) s += ` Shannon was trained on exactly ${M} ideas (${M * R} word-pieces); shorter input works but is less like training.`;
  $("tok-info").textContent = s;
}

async function encode() {
  const { ids } = prepIds();
  if (ids.length < S.meta.arch.R) { $("tok-info").textContent = "Type a little more text."; return; }
  const t0 = performance.now();
  const input = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const out = await S.sess.encoder.run({ input_ids: input });
  const codes = Array.from(out.codes.data, Number);
  const R = S.meta.arch.R;
  S.spans = codes.map((_, j) => S.tok.decode(ids.slice(j * R, (j + 1) * R)));
  S.codes = codes.slice();
  S.origCodes = codes.slice();
  S.sel = -1;
  $("ideas-info").textContent = `${codes.length} codes, ${(performance.now() - t0).toFixed(0)} ms`;
  renderChips();
  ["gen-btn", "abl-null", "abl-shuf", "abl-reset"].forEach((b) => ($(b).disabled = false));
  drawMap();
}

function renderChips() {
  const el = $("chips");
  el.innerHTML = "";
  S.codes.forEach((c, j) => {
    const b = document.createElement("button");
    b.className = "chip" + (c !== S.origCodes[j] ? " edited" : "") + (j === S.sel ? " sel" : "");
    const top = S.dict && S.dict[c] && S.dict[c].s ? S.dict[c].s[0][0] : "";
    const label = c !== S.origCodes[j] ? `was ${S.origCodes[j]} · ${top}` : S.spans[j];
    b.innerHTML = `<b>#${c}</b><span>${esc(label)}</span>`;
    b.title = `slot ${j + 1}: code ${c}`;
    b.onclick = () => { S.sel = j; renderChips(); showCard($("chip-card"), c, j); drawMap(); };
    el.appendChild(b);
  });
}

function setCode(j, c) {
  S.codes[j] = c;
  renderChips();
  showCard($("chip-card"), c, j);
  drawMap();
}

// ------------------------------------------------------------------------------------------ ideas -> text
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function sample(logits, prevIds, o, rand) {
  const V = logits.length, minId = S.meta.arch.min_gen_id;
  const l = new Float32Array(logits);
  for (let i = 0; i < minId; i++) l[i] = -Infinity;
  if (o.rep > 1) for (const i of new Set(prevIds.slice(-64))) l[i] = l[i] > 0 ? l[i] / o.rep : l[i] * o.rep;
  if (o.temp <= 0) { let b = 0; for (let i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; }
  const k = Math.max(1, Math.min(o.topk, V));
  const idx = [];
  for (let i = minId; i < V; i++) {
    if (idx.length < k) { idx.push(i); if (idx.length === k) idx.sort((a, b) => l[b] - l[a]); continue; }
    if (l[i] > l[idx[k - 1]]) {
      let p = k - 1;
      while (p > 0 && l[idx[p - 1]] < l[i]) { idx[p] = idx[p - 1]; p--; }
      idx[p] = i;
    }
  }
  if (idx.length < k) idx.sort((a, b) => l[b] - l[a]);
  const mx = l[idx[0]];
  const w = idx.map((i) => Math.exp((l[i] - mx) / o.temp));
  let tot = w.reduce((a, b) => a + b, 0), r = rand() * tot;
  for (let j = 0; j < idx.length; j++) { r -= w[j]; if (r <= 0) return idx[j]; }
  return idx[idx.length - 1];
}

async function generate() {
  if (S.running || !S.codes) return;
  S.running = true; S.stop = false;
  $("gen-btn").disabled = true; $("stop-btn").disabled = false; $("encode-btn").disabled = true;
  const a = S.meta.arch, L = a.n_dec_layers, H = a.n_heads, T = a.lout, dh = a.head_dim;
  const o = { temp: +$("temp").value, topk: +$("topk").value | 0, rep: +$("rep").value, n: Math.min(T, Math.max(1, +$("len").value | 0)) };
  const rand = rng(+$("seed").value | 0);
  const out = $("output");
  out.classList.remove("muted");
  out.textContent = "";
  try {
    const t0 = performance.now();
    const codes = new ort.Tensor("int64", BigInt64Array.from(S.codes.map(BigInt)), [1, S.codes.length]);
    const cr = await S.sess.ideas.run({ codes });
    let kc = new ort.Tensor("float32", new Float32Array(L * H * T * dh), [L, H, T, dh]);
    let vc = new ort.Tensor("float32", new Float32Array(L * H * T * dh), [L, H, T, dh]);
    let prev = a.bos_id;
    const gen = [];
    for (let t = 0; t < o.n && !S.stop; t++) {
      const r = await S.sess.decoder.run({
        token: new ort.Tensor("int64", BigInt64Array.from([BigInt(prev)]), [1]),
        pos: new ort.Tensor("int64", BigInt64Array.from([BigInt(t)]), [1]),
        k_cache: kc, v_cache: vc, cross_k: cr.cross_k, cross_v: cr.cross_v,
      });
      kc = r.k_cache_out; vc = r.v_cache_out;
      prev = sample(r.logits.data, gen, o, rand);
      gen.push(prev);
      out.textContent = S.tok.decode(gen);
      $("gen-info").textContent = `${gen.length} tokens, ${((performance.now() - t0) / gen.length).toFixed(0)} ms/token`;
      if (t % 4 === 3) await new Promise((res) => setTimeout(res, 0));
    }
    S.lastGen = gen;
  } catch (e) {
    console.error(e);
    out.innerHTML = `<span class="err">${esc(e.message || e)}</span>`;
  } finally {
    S.running = false;
    $("gen-btn").disabled = false; $("stop-btn").disabled = true; $("encode-btn").disabled = false;
  }
}

// ------------------------------------------------------------------------------------------ idea card (shared)
function showCard(el, c, slot = -1) {
  el.classList.remove("hidden");
  const d = S.dict ? S.dict[c] : null;
  const reg = S.map.regions[S.map.region[c]];
  let h = `<h4>Code #${c}</h4><div class="muted small">region: ${esc(reg.label)}${d ? ` · used ${d.n}× in held-out text` : ""}${c === S.stats.modal_code ? " · the most common code" : ""}</div>`;
  if (slot >= 0) h += `<div class="small">Slot ${slot + 1} of ${S.codes.length}. Made from: <b>${esc(S.spans[slot])}</b>${S.codes[slot] !== S.origCodes[slot] ? ` (edited; was #${S.origCodes[slot]})` : ""}</div>`;
  if (!d) h += `<p class="muted">Dictionary still loading…</p>`;
  else if (!d.n) h += `<p class="muted">Never used on the held-out text (a dead or rare code).</p>`;
  else {
    h += `<div class="k">Spans it encoded most</div><ul>${d.s.map(([s, n]) => `<li>${esc(s)} <span class="muted">×${n}</span></li>`).join("")}</ul>`;
    if (d.w && d.w.length) h += `<div class="k">Over-represented words</div><div>${d.w.map(esc).join(", ")}</div>`;
    if (d.x && d.x.length) h += `<div class="k">In context</div><ul class="ctx">${d.x.map(([l, m, r]) => `<li>…${esc(l)} <mark>${esc(m)}</mark> ${esc(r)}…</li>`).join("")}</ul>`;
    if (d.p) h += `<div class="k">Where in the 32 slots</div>${slotBars(d.p)}`;
  }
  h += `<div class="k">Nearest codes${slot >= 0 ? " (tap to swap into this slot)" : ""}</div><div class="nbrs">${S.map.nbr[c].map((n) => `<button data-c="${n}">#${n}</button>`).join("")}</div>`;
  if (slot >= 0 && S.codes[slot] !== S.origCodes[slot]) h += `<div class="row"><button data-undo="1">Undo this slot</button></div>`;
  el.innerHTML = h;
  el.querySelectorAll("button[data-c]").forEach((b) => (b.onclick = () => {
    const n = +b.dataset.c;
    if (slot >= 0) setCode(slot, n); else showDict(n);
  }));
  const u = el.querySelector("button[data-undo]");
  if (u) u.onclick = () => setCode(slot, S.origCodes[slot]);
}

function slotBars(p) {
  const mx = Math.max(...p, 1), w = 22, hgt = 34;
  const bars = p.map((v, i) => `<rect x="${i * (w + 3)}" y="${hgt - (hgt * v) / mx}" width="${w}" height="${(hgt * v) / mx}" fill="var(--accent)" opacity=".75"></rect>`).join("");
  return `<svg viewBox="0 0 ${p.length * (w + 3)} ${hgt + 12}" width="${p.length * (w + 3)}" height="${hgt + 12}" role="img" aria-label="usage by slot group">${bars}<text x="0" y="${hgt + 11}" font-size="9" fill="var(--muted)">slots 1–4</text><text x="${p.length * (w + 3)}" y="${hgt + 11}" font-size="9" text-anchor="end" fill="var(--muted)">29–32</text></svg>`;
}

// ------------------------------------------------------------------------------------------ map
const M = { cv: null, ctx: null, w: 0, h: 0, pad: 18, hover: -1, pick: -1 };
function regionColor(r, n) {
  return `hsl(${(r * 360 / n + 17 * (r % 3)) % 360} 55% ${matchMedia("(prefers-color-scheme: dark)").matches ? 62 : 48}%)`;
}
function initMap() {
  M.cv = $("map-canvas");
  M.ctx = M.cv.getContext("2d");
  const resize = () => {
    const r = M.cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    M.w = r.width; M.h = r.height;
    M.cv.width = Math.round(r.width * dpr); M.cv.height = Math.round(r.height * dpr);
    M.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawMap();
  };
  new ResizeObserver(resize).observe(M.cv);
  const nearest = (ev) => {
    const r = M.cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = -1, bd = 1e9;
    S.map.xy.forEach((p, i) => { const [px, py] = toPx(p); const d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = i; } });
    return bd < 18 ** 2 ? { i: best, x, y } : null;
  };
  M.cv.addEventListener("mousemove", (ev) => { const n = nearest(ev); M.hover = n ? n.i : -1; tip(n); drawMap(); });
  M.cv.addEventListener("mouseleave", () => { M.hover = -1; tip(null); drawMap(); });
  M.cv.addEventListener("click", (ev) => {
    const n = nearest(ev);
    if (!n) return;
    M.pick = n.i; tip(n); drawMap();
    $("map-hint").innerHTML = `Code #${n.i} selected. <a href="#dict" id="map-go">Open in dictionary</a>`;
    $("map-go").onclick = () => showDict(n.i);
  });
  ["show-labels", "show-path"].forEach((id) => $(id).addEventListener("change", drawMap));
}
function toPx([x, y]) { return [M.pad + x * (M.w - 2 * M.pad), M.pad + (1 - y) * (M.h - 2 * M.pad)]; }
function tip(n) {
  const el = $("map-tip");
  if (!n) { el.classList.add("hidden"); return; }
  const c = n.i, d = S.dict ? S.dict[c] : null;
  el.innerHTML = `<b>#${c}</b> · ${esc(S.map.regions[S.map.region[c]].label)}<br>${d && d.s ? d.s.slice(0, 3).map((s) => esc(s[0])).join("<br>") : d ? "unused" : ""}`;
  el.classList.remove("hidden");
  const left = Math.min(n.x + 12, M.w - 270), top = n.y + 12 > M.h - 90 ? n.y - 90 : n.y + 12;
  el.style.left = `${Math.max(4, left)}px`; el.style.top = `${Math.max(4, top)}px`;
}
function drawMap() {
  if (!M.ctx || !S.map || !M.w) return;
  const g = M.ctx, nR = S.map.regions.length, css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, M.w, M.h);
  const rad = M.w < 500 ? 1.6 : 2.2;
  const active = new Set(S.codes || []);
  g.globalAlpha = active.size ? 0.45 : 0.8;
  S.map.xy.forEach((p, i) => {
    const [x, y] = toPx(p);
    g.fillStyle = regionColor(S.map.region[i], nR);
    g.beginPath(); g.arc(x, y, rad, 0, 6.283); g.fill();
  });
  g.globalAlpha = 1;
  const fg = css.getPropertyValue("--fg").trim(), acc = css.getPropertyValue("--accent").trim();
  if ($("show-labels").checked) {
    g.font = `${M.w < 500 ? 10 : 12}px system-ui, sans-serif`; g.textAlign = "center";
    for (const r of S.map.regions) {
      const [x, y] = toPx(r.xy), t = r.label.split(" · ").slice(0, M.w < 500 ? 1 : 2).join(" · ");
      g.lineWidth = 3; g.strokeStyle = css.getPropertyValue("--card").trim(); g.strokeText(t, x, y); g.fillStyle = fg; g.fillText(t, x, y);
    }
  }
  if (S.codes) {
    if ($("show-path").checked) {
      g.strokeStyle = acc; g.lineWidth = 1.2; g.globalAlpha = 0.6; g.beginPath();
      S.codes.forEach((c, j) => { const [x, y] = toPx(S.map.xy[c]); j ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke(); g.globalAlpha = 1;
    }
    S.codes.forEach((c, j) => {
      const [x, y] = toPx(S.map.xy[c]);
      g.strokeStyle = j === S.sel ? fg : acc; g.lineWidth = j === S.sel ? 3 : 2;
      g.beginPath(); g.arc(x, y, j === S.sel ? 7 : 5, 0, 6.283); g.stroke();
    });
  }
  for (const i of [M.hover, M.pick]) if (i >= 0) {
    const [x, y] = toPx(S.map.xy[i]);
    g.strokeStyle = fg; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 6, 0, 6.283); g.stroke();
  }
}

// ------------------------------------------------------------------------------------------ dictionary
function initDict() {
  const draw = () => {
    const q = $("dict-q").value.trim().toLowerCase();
    let ids = S.dict.map((_, i) => i);
    if (/^\d+$/.test(q)) ids = ids.filter((i) => String(i).startsWith(q));
    else if (q) ids = ids.filter((i) => { const d = S.dict[i]; return d.n && ((d.w || []).some((w) => w.includes(q)) || d.s.some(([s]) => s.includes(q))); });
    if ($("dict-sort").value === "n") ids.sort((a, b) => S.dict[b].n - S.dict[a].n);
    const list = $("dict-list");
    list.innerHTML = ids.slice(0, 300).map((i) => {
      const d = S.dict[i];
      const sum = d.n ? d.s.slice(0, 2).map((s) => s[0]).join(" | ") : "unused";
      return `<li data-c="${i}"><b>#${i}</b>${esc(sum)} <small>×${d.n}</small></li>`;
    }).join("") + (ids.length > 300 ? `<li class="muted small">…${ids.length - 300} more; refine the search.</li>` : "") + (ids.length ? "" : `<li class="muted">No codes match.</li>`);
    list.querySelectorAll("li[data-c]").forEach((li) => (li.onclick = () => showDict(+li.dataset.c, false)));
  };
  $("dict-q").addEventListener("input", draw);
  $("dict-sort").addEventListener("change", draw);
  draw();
  showDict(S.stats.modal_code, false);
}
function showDict(c, scroll = true) {
  document.querySelectorAll("#dict-list li.sel").forEach((x) => x.classList.remove("sel"));
  const li = document.querySelector(`#dict-list li[data-c="${c}"]`);
  if (li) li.classList.add("sel");
  showCard($("dict-card"), c);
  M.pick = c; drawMap();
  if (scroll) $("dict").scrollIntoView();
}

// ------------------------------------------------------------------------------------------ stats
function renderStats() {
  const c = S.card.compare, st = S.stats, f = (x, n = 3) => Number(x).toFixed(n);
  const worse = c.shannon.bpb - c.baseline.bpb;
  let h = `<div class="verdict"><b>Shannon is worse than the plain baseline at this size.</b> On the validation split it scores <b>${f(c.shannon.bpb)}</b> bits per byte against <b>${f(c.baseline.bpb)}</b> for a plain encoder-decoder trained with the same data and compute (${f(worse)} BPB worse, about ${f(100 * worse / c.baseline.bpb, 1)}%). Lower is better.</div>`;
  h += `<p>The baseline lets its decoder attend to the encoder's continuous vectors. Shannon forces everything through 32 discrete codes (32 × 12 bits = 384 bits per window), so it pays for the bottleneck. The bet is that discrete ideas are easier to read, edit and reuse, not that they predict text better at this size.</p>`;
  h += `<div class="tbl"><table><thead><tr><th></th><th>Shannon</th><th>Plain baseline</th></tr></thead><tbody>
    <tr><td>Bits per byte (true ideas / encoder memory)</td><td class="num">${f(c.shannon.bpb)}</td><td class="num">${f(c.baseline.bpb)}</td></tr>
    <tr><td>… with another window's ideas (shuffle)</td><td class="num">${f(c.shannon.bpb_shuffle)}</td><td class="num">${f(c.baseline.bpb_shuffle)}</td></tr>
    <tr><td>… with no information (null)</td><td class="num">${f(c.shannon.bpb_null)}</td><td class="num">${f(c.baseline.bpb_null)}</td></tr>
    <tr><td>Gain from the right ideas vs shuffled (G<sub>shuffle</sub>)</td><td class="num">${f(c.shannon.bpb_shuffle - c.shannon.bpb)}</td><td class="num">${f(c.baseline.bpb_shuffle - c.baseline.bpb)}</td></tr>
    <tr><td>Gain vs null (G<sub>null</sub>)</td><td class="num">${f(c.shannon.bpb_null - c.shannon.bpb)}</td><td class="num">${f(c.baseline.bpb_null - c.baseline.bpb)}</td></tr>
    <tr><td>Non-embedding parameters</td><td class="num">${(c.shannon.nonemb_params / 1e6).toFixed(1)}M</td><td class="num">${(c.baseline.nonemb_params / 1e6).toFixed(1)}M</td></tr>
    <tr><td>Training compute</td><td class="num">${c.shannon.train_flops.toExponential(2)} FLOP</td><td class="num">${c.baseline.train_flops.toExponential(2)} FLOP</td></tr>
    </tbody></table></div>
    <p class="muted small">Source: ${esc(c.source)}. BPB counts bits per UTF-8 byte of the lower-cased, tokenised text, so it compares the two arms, not other language models. Both share the same frozen MiniLM encoder (not counted above).</p>`;
  h += `<h3>The ideas carry only a little information</h3>
    <p>On the ${st.heldout_windows.toLocaleString()} held-out validation windows behind this demo's dictionary and presets (scored in fp32 by the build script, so the numbers differ from the table above), Shannon scores ${f(st.bpb.true)} BPB with its own ideas, ${f(st.bpb.shuffle)} with another window's ideas and ${f(st.bpb.null)} with every idea set to the most common code (#${st.modal_code}).
    So the right ideas save just ${f(st.G_shuffle)} bits per byte over the wrong ones. The decoder mostly writes generic, on-topic-ish text; the ideas steer the topic more than the wording.
    The edits and ablation buttons above let you see this for yourself.</p>`;
  h += `<h3>Codebook use</h3><p>${st.codes_used.toLocaleString()} of ${st.K.toLocaleString()} codes appear on the held-out text; code perplexity is ${Math.round(st.code_perplexity).toLocaleString()} (the codebook is well used, not collapsed). Code frequency by rank:</p><div class="chart">${rankChart(st.rank_counts)}</div>`;
  h += `<h3>Export check</h3><p class="muted small">The browser graphs were checked against PyTorch on ${S.meta.export_check.n_windows} windows: same idea codes ${f(100 * S.meta.export_check.shipped_candidate.code_agreement, 0)}% of the time, BPB within ${f(100 * Math.abs(S.meta.export_check.shipped_candidate.bpb_rel_diff), 2)}% (${esc(S.meta.export_check.precision)}).</p>`;
  $("stats-body").innerHTML = h;
}
function rankChart(r) {
  const W = 560, H = 200, p = { l: 40, r: 10, t: 10, b: 30 }, n = r.length, mx = r[0];
  const lx = (i) => p.l + (Math.log10(i + 1) / Math.log10(n)) * (W - p.l - p.r);
  const ly = (v) => p.t + (1 - Math.log10(v + 1) / Math.log10(mx + 1)) * (H - p.t - p.b);
  let d = "";
  for (let i = 0; i < n; i += i < 100 ? 1 : 8) d += `${d ? "L" : "M"}${lx(i).toFixed(1)},${ly(r[i]).toFixed(1)}`;
  const ticksX = [1, 10, 100, 1000, n].map((t) => `<text x="${lx(t - 1)}" y="${H - 10}" text-anchor="middle">${t}</text>`).join("");
  const ticksY = [1, 10, mx].map((t) => `<text x="${p.l - 6}" y="${ly(t) + 4}" text-anchor="end">${t}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Code usage count by rank, log-log"><line x1="${p.l}" y1="${H - p.b}" x2="${W - p.r}" y2="${H - p.b}" stroke="var(--line)"/><line x1="${p.l}" y1="${p.t}" x2="${p.l}" y2="${H - p.b}" stroke="var(--line)"/><path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"/>${ticksX}${ticksY}<text x="${W / 2}" y="${H}" text-anchor="middle">rank (log)</text></svg>`;
}

// ------------------------------------------------------------------------------------------ wiring
$("load-btn").onclick = loadModel;
$("encode-btn").onclick = () => encode().catch((e) => { console.error(e); $("tok-info").innerHTML = `<span class="err">${esc(e.message || e)}</span>`; });
$("gen-btn").onclick = generate;
$("stop-btn").onclick = () => (S.stop = true);
$("input").addEventListener("input", () => { updateTokInfo(); $("preset").value = ""; $("truth-box").classList.add("hidden"); });
$("preset").addEventListener("change", () => {
  const k = $("preset").value;
  if (k === "") { $("truth-box").classList.add("hidden"); return; }
  const p = S.presets[+k];
  $("input").value = p.text;
  $("truth").textContent = p.next;
  $("truth-bpb").textContent = `Shannon's loss on this real continuation: ${p.bpb.toFixed(3)} bits per byte (one of its easiest held-out windows).`;
  $("truth-box").classList.remove("hidden");
  updateTokInfo();
  if (S.sess) encode();
});
$("abl-null").onclick = () => { S.codes = S.codes.map(() => S.stats.modal_code); S.sel = -1; renderChips(); $("chip-card").classList.add("hidden"); drawMap(); };
$("abl-shuf").onclick = () => {
  const r = rng(+$("seed").value | 0);
  for (let i = S.codes.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [S.codes[i], S.codes[j]] = [S.codes[j], S.codes[i]]; }
  S.sel = -1; renderChips(); $("chip-card").classList.add("hidden"); drawMap();
};
$("abl-reset").onclick = () => { S.codes = S.origCodes.slice(); S.sel = -1; renderChips(); $("chip-card").classList.add("hidden"); drawMap(); };
window.shannon = S; // for debugging in the console

boot().catch((e) => { console.error(e); $("model-line").innerHTML = `<span class="err">Failed to start: ${esc(e.message || e)}</span>`; });
