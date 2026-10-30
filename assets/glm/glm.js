// GLM v0 core: graph packs, name linker, MiniLM token encoder and the operator path scorer (KG3b R2).
// Pure module (no DOM): runs in the page's Web Worker and in Node for the parity check (tools/glm_parity/parity.mjs).
// Pack format: PACK_FORMAT.md.

import { WordPiece } from "../qa/tokenizers.js";

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

// ------------------------------------------------------------------ small dense linear algebra (row-major D x D)
function matmul(A, B, D) {
  const C = new Float64Array(D * D);
  for (let i = 0; i < D; i++)
    for (let k = 0; k < D; k++) {
      const a = A[i * D + k];
      if (a === 0) continue;
      for (let j = 0; j < D; j++) C[i * D + j] += a * B[k * D + j];
    }
  return C;
}

// matrix exponential: scaling and squaring with a Taylor series (float64)
export function expm(A, D) {
  let nrm = 0;
  for (let i = 0; i < D; i++) {
    let s = 0;
    for (let j = 0; j < D; j++) s += Math.abs(A[i * D + j]);
    nrm = Math.max(nrm, s);
  }
  const sq = nrm > 0.5 ? Math.ceil(Math.log2(nrm / 0.5)) : 0;
  const X = Float64Array.from(A, v => v / 2 ** sq);
  let E = new Float64Array(D * D), term = new Float64Array(D * D);
  for (let i = 0; i < D; i++) { E[i * D + i] = 1; term[i * D + i] = 1; }
  for (let k = 1; k <= 18; k++) {
    term = matmul(term, X, D);
    for (let i = 0; i < D * D; i++) { term[i] /= k; E[i] += term[i]; }
  }
  for (let s = 0; s < sq; s++) E = matmul(E, E, D);
  return E;
}

const normed = v => {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float64Array.from(v, x => x / s);
};

// ------------------------------------------------------------------ name normalisation for the linker
const MARKS = /\p{Mn}/gu;
const WORD = /[\p{L}\p{N}]+/gu;
export function words(text) {
  const out = [];
  for (const m of text.matchAll(WORD)) out.push({ w: m[0].normalize("NFD").replace(MARKS, "").toLowerCase(), a: m.index, b: m.index + m[0].length });
  return out;
}
const normName = s => words(s).map(x => x.w).join(" ");
const STOP = new Set("a an the of in on to by for and or is are was were be what which who whom whose when where how did do does film films movie movies".split(" "));

// ------------------------------------------------------------------ pack
export class Pack {
  // manifest: parsed manifest.json; files: {entities: string, graph: ArrayBuffer, weights: ArrayBuffer}
  constructor(manifest, files) {
    const t0 = now();
    this.m = manifest;
    this.id = manifest.id;
    this.names = files.entities.split("\n");
    if (this.names[this.names.length - 1] === "") this.names.pop();
    const N = this.n = this.names.length;
    const R = manifest.relations.length, NE = this.NE = 2 * R;
    this.edgeNames = [];
    this.edgeLabels = [];
    for (const r of manifest.relations) {
      this.edgeNames.push(r.id, r.id + manifest.inverse_suffix);
      this.edgeLabels.push(r.label, r.inverse_label || r.label + " (inverse)");
    }
    // graph -> CSR per edge type (e = 2r head->tail, 2r+1 tail->head)
    const T = manifest.counts.triples, g = files.graph;
    const H = new Uint32Array(g, 0, T), Tl = new Uint32Array(g, 4 * T, T);
    const Rl = manifest.files.graph.rel_dtype === "u16" ? new Uint16Array(g, 8 * T, T) : new Uint8Array(g, 8 * T, T);
    this.off = []; this.to = [];
    const cnt = Array.from({ length: NE }, () => new Uint32Array(N + 1));
    for (let i = 0; i < T; i++) { cnt[2 * Rl[i]][H[i] + 1]++; cnt[2 * Rl[i] + 1][Tl[i] + 1]++; }
    for (let e = 0; e < NE; e++) {
      const o = cnt[e];
      for (let i = 0; i < N; i++) o[i + 1] += o[i];
      this.off.push(o); this.to.push(new Uint32Array(o[N]));
    }
    const fill = cnt.map(o => o.slice(0, N));
    for (let i = 0; i < T; i++) {
      const r = Rl[i];
      this.to[2 * r][fill[2 * r][H[i]]++] = Tl[i];
      this.to[2 * r + 1][fill[2 * r + 1][Tl[i]]++] = H[i];
    }
    // degree and "weak" entities (every edge is a weak relation, e.g. MetaQA tags)
    const weakR = new Set((manifest.linker.weak_relations || []).map(id => manifest.relations.findIndex(r => r.id === id)));
    this.deg = new Uint32Array(N); const strong = new Uint8Array(N);
    for (let i = 0; i < T; i++) {
      this.deg[H[i]]++; this.deg[Tl[i]]++;
      if (!weakR.has(Rl[i])) { strong[H[i]] = 1; strong[Tl[i]] = 1; }
    }
    this.strong = strong;
    // name index: normalised name -> entity ids (aliases optional)
    this.index = new Map(); this.maxWords = 1;
    const add = (name, id) => {
      const k = normName(name);
      if (!k || STOP.has(k)) return;
      let l = this.index.get(k);
      if (!l) this.index.set(k, (l = []));
      l.push(id);
      this.maxWords = Math.max(this.maxWords, k.split(" ").length);
    };
    this.names.forEach((nm, id) => add(nm, id));
    if (files.aliases) for (const line of files.aliases.split("\n")) {
      const [al, id] = line.split("\t");
      if (al && id) add(al, +id);
    }
    this.maxWords = Math.min(this.maxWords, 16);
    // scorer weights
    const W = new Float32Array(files.weights), L = manifest.files.weights.layout, sc = manifest.scorer;
    const get = k => W.subarray(L[k].offset, L[k].offset + L[k].shape.reduce((a, b) => a * b, 1));
    this.mu = get("mu"); this.sd = get("sd"); this.W = get("W"); this.b = get("b");
    this.D = sc.dim; this.tau = sc.tau; this.C = sc.tagger_classes; this.dTok = sc.tok_dim;
    const DD = this.D * this.D;
    this.Le = []; this.Me = [];
    const le = get("Le"), me = get("Me");
    for (let e = 0; e < NE; e++) { this.Le.push(Float64Array.from(le.subarray(e * DD, (e + 1) * DD))); this.Me.push(Float64Array.from(me.subarray(e * DD, (e + 1) * DD))); }
    this.pathVec = new Map(); this.prod = new Map();
    this.buildMs = now() - t0;
  }

  // ---- linker: longest run of words that is an entity name; ties -> strong entity, then higher degree.
  // An explicit [Name] in the question wins.
  link(q) {
    const br = /\[([^\]]+)\]/.exec(q);
    if (br) {
      const ids = this.index.get(normName(br[1]));
      if (ids) return this._pick(ids, br.index, br.index + br[0].length, br[1], 1e9);
    }
    const ws = words(q);
    let best = null;
    for (let i = 0; i < ws.length; i++) {
      let k = "";
      for (let j = i; j < Math.min(ws.length, i + this.maxWords); j++) {
        k = j === i ? ws[j].w : k + " " + ws[j].w;
        const ids = this.index.get(k);
        if (!ids) continue;
        const c = this._pick(ids, ws[i].a, ws[j].b, q.slice(ws[i].a, ws[j].b), k.length);
        const tail = /[^\p{L}\p{N}]+$/u.exec(c.name);  // names ending in punctuation ("That's Entertainment!")
        if (tail && q.startsWith(tail[0], c.b)) { c.b += tail[0].length; c.span = q.slice(c.a, c.b); }
        if (!best || c.rank > best.rank) best = c;
      }
    }
    return best;
  }

  _pick(ids, a, b, span, len) {
    let id = ids[0];
    for (const x of ids) if (this.strong[x] > this.strong[id] || (this.strong[x] === this.strong[id] && this.deg[x] > this.deg[id])) id = x;
    // rank: strong entities first, then longer spans, then degree
    const rank = this.strong[id] * 1e12 + len * 1e6 + Math.min(this.deg[id], 999999);
    return { pack: this.id, id, name: this.names[id], a, b, span, rank, ambiguous: ids.length };
  }

  // ---- graph walking (sets with path counts; topic removed after every step = no-revisit)
  step(front, e, topic) {
    const off = this.off[e], to = this.to[e], out = new Map();
    for (const [v, c] of front) {
      for (let k = off[v]; k < off[v + 1]; k++) {
        const u = to[k];
        if (u === topic) continue;
        out.set(u, (out.get(u) || 0) + c);
      }
    }
    return out;
  }

  hasOut(v, e) { return this.off[e][v + 1] > this.off[e][v]; }

  // every executable 1..maxHops path from the topic (same rule as kg3b.candidates)
  candidates(t) {
    const NE = this.NE, out = [];
    const x0 = new Map([[t, 1]]);
    for (let a = 0; a < NE; a++) {
      const x1 = this.step(x0, a, t);
      if (!x1.size) continue;
      out.push([a]);
      for (let b = 0; b < NE; b++) {
        const x2 = this.stepSet(x1, b, t);
        if (!x2.size) continue;
        out.push([a, b]);
        const av = new Uint8Array(NE);
        let left = NE;
        for (const v of x2) {
          for (let c = 0; c < NE; c++) if (!av[c] && this.hasOut(v, c)) { av[c] = 1; left--; }
          if (!left) break;
        }
        for (let c = 0; c < NE; c++) if (av[c]) out.push([a, b, c]);
      }
    }
    return out;
  }

  stepSet(front, e, topic) {
    const off = this.off[e], to = this.to[e], out = new Set();
    for (const v of front.keys()) for (let k = off[v]; k < off[v + 1]; k++) if (to[k] !== topic) out.add(to[k]);
    return out;
  }

  frontier(t, path) {
    let x = new Map([[t, 1]]);
    for (const e of path) { x = this.step(x, e, t); if (!x.size) break; }
    return x;
  }

  // ---- scorer
  // tok: Float32Array (n x dTok) MiniLM token vectors of the masked question (no [CLS]/[SEP])
  questionOperator(tok, n) {
    const d = this.dTok, C = this.C, NE = this.NE, counts = new Float64Array(NE), z = new Float64Array(C);
    const tags = [];
    for (let t = 0; t < n; t++) {
      let mx = -Infinity;
      for (let c = 0; c < C; c++) {
        let s = this.b[c];
        const w = c * d;
        for (let i = 0; i < d; i++) s += this.W[w + i] * ((tok[t * d + i] - this.mu[i]) / this.sd[i]);
        z[c] = s; if (s > mx) mx = s;
      }
      let S = 0;
      for (let c = 0; c < C; c++) { z[c] = Math.exp(z[c] - mx); S += z[c]; }
      const p = Array.from(z, v => v / S);
      for (let e = 0; e < NE; e++) counts[e] += p[e];
      tags.push(p);
    }
    const DD = this.D * this.D, Q = new Float64Array(DD);
    for (let e = 0; e < NE; e++) if (counts[e]) for (let i = 0; i < DD; i++) Q[i] += counts[e] * this.Le[e][i];
    return { counts, tags, q: normed(expm(Q, this.D)) };
  }

  pathProduct(p) {
    if (p.length === 1) return this.Me[p[0]];
    const k = p.join(",");
    let M = this.prod.get(k);
    if (!M) { M = matmul(this.pathProduct(p.slice(0, -1)), this.Me[p[p.length - 1]], this.D); this.prod.set(k, M); }
    return M;
  }

  pathVector(p) {
    const k = p.join(",");
    let v = this.pathVec.get(k);
    if (!v) { v = normed(this.pathProduct(p)); this.pathVec.set(k, v); }
    return v;
  }

  // full answer for a linked topic. Returns the chosen path, answers and ranked alternatives.
  answer(tok, n, topic, opts = {}) {
    const tm = {}, sc = this.m.scorer;
    let t0 = now();
    const qo = this.questionOperator(tok, n);
    tm.tagger_ms = now() - t0; t0 = now();
    const cands = this.candidates(topic);
    tm.paths_ms = now() - t0; t0 = now();
    const scored = cands.map(p => {
      const v = this.pathVector(p);
      let c = 0;
      for (let i = 0; i < v.length; i++) c += v[i] * qo.q[i];
      return { path: p, score: this.tau * c };
    });
    scored.sort((x, y) => y.score - x.score);
    tm.score_ms = now() - t0; t0 = now();
    let chosen = null, rank = -1, front = null;
    for (let i = 0; i < Math.min(sc.top_k_walk, scored.length); i++) {
      const f = this.frontier(topic, scored[i].path);
      if (f.size) { chosen = scored[i]; rank = i; front = f; break; }
    }
    tm.walk_ms = now() - t0;
    if (!chosen) return { ok: false, reason: "no path from this entity leads anywhere", n_cands: cands.length, timings: tm };
    const ranked = [...front].sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    const top = ranked[0][1];
    const alts = [];
    for (let i = 0; i < scored.length && alts.length < (opts.alts ?? 3); i++) {
      if (i === rank) continue;
      const f = this.frontier(topic, scored[i].path);
      if (!f.size) continue;
      const r = [...f].sort((x, y) => y[1] - x[1] || x[0] - y[0]);
      alts.push({ path: scored[i].path.map(e => this.edgeNames[e]), labels: scored[i].path.map(e => this.edgeLabels[e]), score: scored[i].score,
        answers: r.filter(x => x[1] === r[0][1]).slice(0, 5).map(x => this.names[x[0]]), n: f.size });
    }
    // intermediate nodes along the chosen path (for drawing)
    const hops = [];
    let x = new Map([[topic, 1]]);
    for (const e of chosen.path) {
      x = this.step(x, e, topic);
      const r = [...x].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
      hops.push({ edge: this.edgeNames[e], label: this.edgeLabels[e], n: x.size, sample: r.slice(0, 4).map(y => this.names[y[0]]) });
    }
    return {
      ok: true, topic: this.names[topic], path: chosen.path.map(e => this.edgeNames[e]), score: chosen.score, walked_rank: rank,
      answers: ranked.filter(y => y[1] === top).map(y => this.names[y[0]]),
      others: ranked.filter(y => y[1] !== top).slice(0, 10).map(y => [this.names[y[0]], y[1]]),
      n_frontier: front.size, n_cands: cands.length, hops, alternatives: alts,
      top5: scored.slice(0, 5).map(s => [s.path.map(e => this.edgeNames[e]), s.score]),
      relation_mass: Array.from(qo.counts).map((v, e) => [this.edgeNames[e], v]).filter(x => x[1] > 0.05).sort((a, b) => b[1] - a[1]),
      timings: tm,
    };
  }
}

// ------------------------------------------------------------------ encoder (MiniLM token vectors through onnxruntime)
export class Encoder {
  constructor(ort, vocabText) { this.ort = ort; this.wp = new WordPiece(vocabText); }
  async init(modelBytes, opts = {}) { this.s = await this.ort.InferenceSession.create(modelBytes, opts); return this; }
  // -> {tok: Float32Array(n*384), n}  (positions 1..n of [CLS] text [SEP], [SEP] dropped; max 32 tokens like the Python prep)
  async tokens(text, maxTok = 32) {
    const e = this.wp.encode(text);
    const body = e.ids.slice(0, maxTok);
    const ids = [this.wp.cls, ...body, this.wp.sep], L = ids.length;
    const big = a => new this.ort.Tensor("int64", BigInt64Array.from(a.map(BigInt)), [1, L]);
    const all = { input_ids: big(ids), attention_mask: big(ids.map(() => 1)), token_type_ids: big(ids.map(() => 0)) };
    const f = {};
    for (const k of this.s.inputNames) f[k] = all[k];
    const r = await this.s.run(f);
    const h = r[this.s.outputNames.includes("last_hidden_state") ? "last_hidden_state" : this.s.outputNames[0]].data;
    const d = h.length / L, n = body.length;
    return { tok: h.slice(d, d * (n + 1)), n, d };
  }
}

// ------------------------------------------------------------------ GLM: encoder + any number of loaded packs
export class GLM {
  constructor(encoder) { this.enc = encoder; this.packs = new Map(); }
  load(pack) { this.packs.set(pack.id, pack); }
  unload(id) { return this.packs.delete(id); }

  async ask(question) {
    const T0 = now(), tm = {};
    if (!this.packs.size) return { ok: false, reason: "No knowledge pack is loaded. Load one to ask questions.", timings: tm };
    let t0 = now(), link = null, pack = null;
    for (const p of this.packs.values()) {
      const l = p.link(question);
      if (l && (!link || l.rank > link.rank)) { link = l; pack = p; }
    }
    tm.link_ms = now() - t0;
    if (!link) return { ok: false, reason: "I could not find a name from the loaded packs in the question.", timings: tm };
    const masked = (question.slice(0, link.a) + pack.m.scorer.mask_token + question.slice(link.b)).replace(/\[|\]/g, "");
    t0 = now();
    const { tok, n } = await this.enc.tokens(masked, pack.m.scorer.max_tokens);
    tm.encoder_ms = now() - t0;
    const r = pack.answer(tok, n, link.id);
    Object.assign(tm, r.timings);
    tm.total_ms = now() - T0;
    return { ...r, pack: pack.id, link, masked, timings: tm };
  }
}
