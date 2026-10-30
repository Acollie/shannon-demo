// GLM v0 core: graph packs, name linker, MiniLM token encoder and the operator path scorer (KG3b R2).
// Pure module (no DOM): runs on the page and in Node for the parity check (tools/glm_parity/parity.mjs).
// A conversation pack (ConvPack, taught facts) overlays a loaded pack without touching its arrays; with no overlay the
// code paths and results are the same as before.
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
export const normName = s => words(s).map(x => x.w).join(" ");
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
      const ids = this.lookup(normName(br[1]));
      if (ids) return this._pick(ids, br.index, br.index + br[0].length, br[1], 1e9);
    }
    const ws = words(q);
    let best = null;
    for (let i = 0; i < ws.length; i++) {
      let k = "";
      const mw = this.ov ? Math.max(this.maxWords, this.ov.maxWords) : this.maxWords;
      for (let j = i; j < Math.min(ws.length, i + mw); j++) {
        k = j === i ? ws[j].w : k + " " + ws[j].w;
        const ids = this.lookup(k);
        if (!ids) continue;
        const c = this._pick(ids, ws[i].a, ws[j].b, q.slice(ws[i].a, ws[j].b), k.length);
        const tail = /[^\p{L}\p{N}]+$/u.exec(c.name);  // names ending in punctuation ("That's Entertainment!")
        if (tail && q.startsWith(tail[0], c.b)) { c.b += tail[0].length; c.span = q.slice(c.a, c.b); }
        if (!best || c.rank > best.rank) best = c;
      }
    }
    return best;
  }

  lookup(k) { return this.index.get(k) || (this.ov ? this.ov.index.get(k) : undefined); }
  st(x) { return x < this.n ? this.strong[x] : 1; }
  dg(x) { return x < this.n ? this.deg[x] : (this.ov.deg[x - this.n] || 0); }
  nm(x) { return x < this.n ? this.names[x] : this.ov.names[x - this.n]; }

  _pick(ids, a, b, span, len) {
    let id = ids[0];
    for (const x of ids) if (this.st(x) > this.st(id) || (this.st(x) === this.st(id) && this.dg(x) > this.dg(id))) id = x;
    // rank: strong entities first, then longer spans, then degree
    const rank = this.st(id) * 1e12 + len * 1e6 + Math.min(this.dg(id), 999999);
    return { pack: this.id, id, name: this.nm(id), a, b, span, rank, ambiguous: ids.length };
  }

  // ---- graph walking (sets with path counts; topic removed after every step = no-revisit)
  step(front, e, topic) {
    const off = this.off[e], to = this.to[e], out = new Map(), N = this.n, ov = this.ov ? this.ov.adj[e] : null;
    for (const [v, c] of front) {
      if (v < N) for (let k = off[v]; k < off[v + 1]; k++) {
        const u = to[k];
        if (u === topic) continue;
        out.set(u, (out.get(u) || 0) + c);
      }
      const l = ov && ov.get(v);
      if (l) for (const u of l) if (u !== topic) out.set(u, (out.get(u) || 0) + c);
    }
    return out;
  }

  hasOut(v, e) { return (v < this.n && this.off[e][v + 1] > this.off[e][v]) || (!!this.ov && this.ov.adj[e].has(v)); }

  // neighbours of v along edge type e (base pack, then taught facts)
  nbrs(v, e) {
    const base = v < this.n ? Array.from(this.to[e].subarray(this.off[e][v], this.off[e][v + 1])) : [];
    const l = this.ov && this.ov.adj[e].get(v);
    return l ? base.concat(l) : base;
  }
  isTaught(u, e, v) { const l = this.ov && this.ov.adj[e].get(u); return !!l && l.includes(v); }

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
    const N = this.n, ov = this.ov ? this.ov.adj[e] : null;
    for (const v of front.keys()) {
      if (v < N) for (let k = off[v]; k < off[v + 1]; k++) if (to[k] !== topic) out.add(to[k]);
      const l = ov && ov.get(v);
      if (l) for (const u of l) if (u !== topic) out.add(u);
    }
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
        answers: r.filter(x => x[1] === r[0][1]).slice(0, 5).map(x => this.nm(x[0])), n: f.size, path_e: scored[i].path });
    }
    // intermediate nodes along the chosen path (for drawing)
    const hops = [];
    let x = new Map([[topic, 1]]);
    for (const e of chosen.path) {
      x = this.step(x, e, topic);
      const r = [...x].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
      hops.push({ edge: this.edgeNames[e], label: this.edgeLabels[e], n: x.size, sample: r.slice(0, 4).map(y => this.nm(y[0])) });
    }
    return {
      ok: true, topic: this.nm(topic), topic_id: topic, path: chosen.path.map(e => this.edgeNames[e]), path_e: chosen.path, score: chosen.score, walked_rank: rank,
      answers: ranked.filter(y => y[1] === top).map(y => this.nm(y[0])),
      others: ranked.filter(y => y[1] !== top).slice(0, 10).map(y => [this.nm(y[0]), y[1]]),
      n_frontier: front.size, n_cands: cands.length, hops, alternatives: alts,
      top5: scored.slice(0, 5).map(s => [s.path.map(e => this.edgeNames[e]), s.score]),
      relation_mass: Array.from(qo.counts).map((v, e) => [this.edgeNames[e], v]).filter(x => x[1] > 0.05).sort((a, b) => b[1] - a[1]),
      timings: tm,
    };
  }
}

// ------------------------------------------------------------------ explanation helpers (drawing, facts, hashing)
// Local graph around the topic for the "thinking graph": the chosen path, the alternatives, and a few other neighbours of
// the topic. Big frontiers are sampled backwards from the reached nodes so every shown node connects to the topic.
Pack.prototype.subgraph = function (topic, paths, { cap = 150, lim = [12, 4], ctx = 3, ctxMax = 24 } = {}) {
  const N = this.n, nodes = new Map(), edges = [], ekeys = new Set(), stats = [];
  const RANK = { context: 0, alt: 1, chosen: 2 };
  const addNode = (id, depth, kind) => {
    let n = nodes.get(id);
    if (!n) {
      if (nodes.size >= cap) return null;
      n = { id, name: this.nm(id), depth, kind, isNew: id >= N, answer: false, taught: false };
      nodes.set(id, n);
    } else if (RANK[kind] > RANK[n.kind]) n.kind = kind;
    return n;
  };
  const addEdge = (a, b, e, kind, hop, pi) => {
    const k = a + ">" + e + ">" + b;
    if (ekeys.has(k) || !nodes.has(a) || !nodes.has(b)) return;
    ekeys.add(k);
    const taught = this.isTaught(a, e, b);
    edges.push({ a, b, e, label: this.edgeLabels[e], kind, hop, pi, taught });
    if (taught) { nodes.get(a).taught = true; nodes.get(b).taught = true; }
  };
  addNode(topic, 0, "chosen").topic = true;
  const reach = new Set([topic]);
  let answerNames = [];
  paths.forEach(({ e: path, kind }, pi) => {
    const F = [new Map([[topic, 1]])];
    for (const e of path) F.push(this.step(F[F.length - 1], e, topic));
    for (const f of F) for (const v of f.keys()) reach.add(v);
    const k = path.length, L = kind === "chosen" ? lim[0] : lim[1];
    // nodes reached through a taught fact somewhere along the path
    const T = [new Set()];
    for (let i = 1; i <= k; i++) {
      const t = new Set(), e = path[i - 1];
      if (this.ov) for (const u of F[i - 1].keys()) for (const v of this.nbrs(u, e))
        if (F[i].has(v) && (T[i - 1].has(u) || this.isTaught(u, e, v))) t.add(v);
      T.push(t);
    }
    const last = [...F[k]].sort((x, y) => (T[k].has(y[0]) - T[k].has(x[0])) || y[1] - x[1] || x[0] - y[0]);
    const top = last.length ? Math.max(...last.map(x => x[1])) : 0;
    const layers = new Array(k + 1);
    layers[k] = last.slice(0, L).map(x => x[0]);
    const links = [];
    for (let i = k; i >= 1; i--) {
      // keep each column at most L nodes: a node joins only through a parent already kept or while there is room
      const prev = F[i - 1], e = path[i - 1], picked = new Set(), kept = [];
      for (const v of layers[i]) {
        const ns = this.nbrs(v, e ^ 1).filter(u => prev.has(u));
        ns.sort((x, y) => (picked.has(y) - picked.has(x)) || (T[i - 1].has(y) - T[i - 1].has(x)) || this.isTaught(y, e, v) - this.isTaught(x, e, v));
        let got = 0;
        for (const u of ns) {
          if (got >= (i === k ? 2 : 1)) break;
          if (!picked.has(u) && picked.size >= L) continue;
          picked.add(u); links.push([u, v, e, i - 1]); got++;
        }
        if (got) kept.push(v);
      }
      layers[i] = kept;
      layers[i - 1] = [...picked];
    }
    for (let i = 1; i <= k; i++) for (const id of layers[i]) {
      const n = addNode(id, i, kind);
      if (n && kind === "chosen" && i === k && F[k].get(id) === top) n.answer = true;
    }
    for (const [u, v, e, hop] of links) addEdge(u, v, e, kind, hop, pi);
    if (kind === "chosen") {
      for (let i = 1; i <= k; i++) stats.push({ hop: i, label: this.edgeLabels[path[i - 1]], n: F[i].size, shown: layers[i].filter(id => nodes.has(id)).length });
      // answers that a taught fact leads straight into (last step), or that are new names
      const viaLast = x => x >= N || (this.ov && this.nbrs(x, path[k - 1] ^ 1).some(u => F[k - 1].has(u) && this.isTaught(u, path[k - 1], x)));
      answerNames = last.filter(x => x[1] === top && T[k].has(x[0]) && viaLast(x[0])).map(x => this.nm(x[0]));
    }
  });
  // other neighbours of the topic: what the walker could have followed but did not
  const firsts = new Set(paths.map(p => p.e[0]));
  let nc = 0;
  for (let e = 0; e < this.NE && nc < ctxMax; e++) {
    if (firsts.has(e)) continue;
    const ns = this.nbrs(topic, e);
    ns.sort((x, y) => this.isTaught(topic, e, y) - this.isTaught(topic, e, x));
    for (const v of ns.slice(0, ctx)) { if (v === topic || nc >= ctxMax) continue; if (!nodes.has(v)) nc++; addNode(v, 1, "context"); addEdge(topic, v, e, "context", 0, -1); }
  }
  return { nodes: [...nodes.values()], edges, stats, reached: reach.size, hidden: Math.max(0, reach.size - nodes.size), taught_answers: answerNames };
};

// name suggestions for the teach form: word-start matches, strong entities and higher degree first
Pack.prototype.suggest = function (text, k = 8) {
  const q = normName(text);
  if (!q) return [];
  if (!this._norm) this._norm = this.names.map(normName);
  const hits = [];
  const consider = (n, id) => { if (n.startsWith(q) || n.includes(" " + q)) hits.push([id, (n === q ? 4 : n.startsWith(q) ? 2 : 0) + this.st(id)]); };
  for (let i = 0; i < this._norm.length; i++) consider(this._norm[i], i);
  if (this.ov) this.ov.names.forEach((nm, i) => consider(normName(nm), this.n + i));
  hits.sort((x, y) => y[1] - x[1] || this.dg(y[0]) - this.dg(x[0]));
  const seen = new Set(), out = [];
  for (const [id] of hits) { const nm = this.nm(id); if (seen.has(nm)) continue; seen.add(nm); out.push({ id, name: nm, deg: this.dg(id) }); if (out.length >= k) break; }
  return out;
};

// facts about one entity, grouped by edge type
Pack.prototype.facts = function (id, per = 8) {
  const out = [];
  for (let e = 0; e < this.NE; e++) {
    const ns = this.nbrs(id, e);
    if (!ns.length) continue;
    ns.sort((x, y) => this.isTaught(id, e, y) - this.isTaught(id, e, x) || this.dg(y) - this.dg(x));
    out.push({ label: this.edgeLabels[e], n: ns.length, items: ns.slice(0, per).map(v => ({ id: v, name: this.nm(v), taught: this.isTaught(id, e, v) })) });
  }
  return { id, name: this.nm(id), isNew: id >= this.n, groups: out };
};

// the base pack's graph as it sits in memory (names + every adjacency array); hashing it shows taught facts never touch it
Pack.prototype.digestBytes = function () {
  const parts = [new TextEncoder().encode(this.names.join("\n"))];
  for (let e = 0; e < this.NE; e++) parts.push(new Uint8Array(this.off[e].buffer, this.off[e].byteOffset, this.off[e].byteLength), new Uint8Array(this.to[e].buffer, this.to[e].byteOffset, this.to[e].byteLength));
  const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

// ------------------------------------------------------------------ conversation pack: facts taught during this visit
// Kept apart from the base pack: base arrays are never written. attach() sets base.ov (an overlay the walker reads);
// detach() removes it, and the base pack is exactly as it was.
export class ConvPack {
  constructor(base) { this.base = base; this.facts = []; }
  // name -> existing entity id (best match by the linker's rule) or null for a new entity
  existing(name) {
    const ids = this.base.index.get(normName(name));
    return ids ? this.base._pick(ids, 0, 0, name, 0).id : null;
  }
  // subject, edge type (forward 2r or inverse 2r+1), object; returns {ok, reason?}
  add(s, e, o) {
    s = String(s || "").trim(); o = String(o || "").trim();
    if (!normName(s) || !normName(o)) return { ok: false, reason: "Both names need at least one letter or digit." };
    if (normName(s) === normName(o)) return { ok: false, reason: "Subject and object are the same." };
    if (STOP.has(normName(s)) || STOP.has(normName(o))) return { ok: false, reason: "That name is too common a word to look up." };
    let h = s, t = o, r = e >> 1;
    if (e & 1) { h = o; t = s; }
    const key = normName(h) + "|" + r + "|" + normName(t);
    if (this.facts.some(f => f.key === key)) return { ok: false, reason: "That fact is already in the conversation pack." };
    const hi = this.existing(h), ti = this.existing(t);
    if (hi != null && ti != null && this.base.nbrs(hi, 2 * r).includes(ti)) return { ok: false, reason: `The ${this.base.id} pack already has that fact.` };
    this.facts.push({ h, r, t, key, hNew: hi == null, tNew: ti == null });
    return { ok: true };
  }
  remove(i) { this.facts.splice(i, 1); }
  build() {
    const B = this.base, NE = B.NE, ov = { names: [], index: new Map(), adj: Array.from({ length: NE }, () => new Map()), deg: [], maxWords: 1 };
    const idOf = name => {
      const x = this.existing(name);
      if (x != null) return x;
      const k = normName(name);
      let l = ov.index.get(k);
      if (!l) { l = [B.n + ov.names.length]; ov.index.set(k, l); ov.names.push(name); ov.deg.push(0); ov.maxWords = Math.max(ov.maxWords, k.split(" ").length); }
      return l[0];
    };
    const push = (e, a, b) => { let l = ov.adj[e].get(a); if (!l) ov.adj[e].set(a, (l = [])); l.push(b); if (a >= B.n) ov.deg[a - B.n]++; };
    for (const f of this.facts) { const a = idOf(f.h), b = idOf(f.t); push(2 * f.r, a, b); push(2 * f.r + 1, b, a); }
    return ov;
  }
  attach() { this.base.ov = this.facts.length ? this.build() : null; return this; }
  detach() { this.base.ov = null; }
  text() { return this.facts.map(f => `${f.h}\t${this.base.m.relations[f.r].id}\t${f.t}`).join("\n"); }
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
  // lean: ONE pass gives both the reader's token vectors (as tokens()) and the linker's sentence vector (as sent()) when
  // the question fits in maxTok word pieces, since then both see the same [CLS] text [SEP] input; otherwise two passes.
  async both(text, maxTok = 32, maxLen = 64) {
    const e = this.wp.encode(text);
    if (e.ids.length > maxTok || e.ids.length > maxLen - 2) {
      const t = await this.tokens(text, maxTok), s = (await this.sent([text], maxLen))[0];
      return { ...t, sent: s, passes: 2 };
    }
    const ids = [this.wp.cls, ...e.ids, this.wp.sep], L = ids.length;
    const big = a => new this.ort.Tensor("int64", BigInt64Array.from(a.map(BigInt)), [1, L]);
    const all = { input_ids: big(ids), attention_mask: big(ids.map(() => 1)), token_type_ids: big(ids.map(() => 0)) };
    const f = {};
    for (const k of this.s.inputNames) f[k] = all[k];
    const r = await this.s.run(f);
    const h = r[this.s.outputNames.includes("last_hidden_state") ? "last_hidden_state" : this.s.outputNames[0]].data;
    const d = h.length / L, n = e.ids.length;
    return { tok: h.slice(d, d * (n + 1)), n, d, sent: pool(h, 0, L, d), passes: 1 };
  }
  // -> Float32Array(384)[]: L2-normalised mean-pooled sentence vectors ([CLS] text [SEP], at most maxLen tokens), like
  // kg15.encode.sent / sentence-transformers all-MiniLM-L6-v2. lean: texts are grouped by token length so no batch is
  // padded (less work, and a text's vector does not depend on which other texts share its batch, so it can be cached).
  async sent(texts, maxLen = 64, batch = 32) {
    const out = new Array(texts.length), byLen = new Map();
    texts.forEach((t, k) => {
      const x = [this.wp.cls, ...this.wp.encode(t).ids.slice(0, maxLen - 2), this.wp.sep];
      let g = byLen.get(x.length); if (!g) byLen.set(x.length, (g = [])); g.push([k, x]);
    });
    for (const [L, g] of byLen) for (let s = 0; s < g.length; s += batch) {
      const part = g.slice(s, s + batch), B = part.length;
      const ids = new BigInt64Array(B * L), am = new BigInt64Array(B * L).fill(1n), tt = new BigInt64Array(B * L);
      part.forEach(([, x], b) => x.forEach((v, j) => { ids[b * L + j] = BigInt(v); }));
      const all = { input_ids: new this.ort.Tensor("int64", ids, [B, L]), attention_mask: new this.ort.Tensor("int64", am, [B, L]), token_type_ids: new this.ort.Tensor("int64", tt, [B, L]) };
      const f = {};
      for (const k of this.s.inputNames) f[k] = all[k];
      const r = await this.s.run(f);
      const h = r[this.s.outputNames.includes("last_hidden_state") ? "last_hidden_state" : this.s.outputNames[0]].data;
      const d = h.length / (B * L);
      part.forEach(([k], b) => { out[k] = pool(h, b * L, L, d); });
    }
    return out;
  }
}

// mean of rows [r0, r0 + L) of h (d wide), L2-normalised -> Float32Array(d)
function pool(h, r0, L, d) {
  const v = new Float64Array(d);
  for (let j = 0; j < L; j++) { const o = (r0 + j) * d; for (let i = 0; i < d; i++) v[i] += h[o + i]; }
  let nrm = 0;
  for (let i = 0; i < d; i++) { v[i] /= L; nrm += v[i] * v[i]; }
  nrm = Math.max(Math.sqrt(nrm), 1e-12);
  return Float32Array.from(v, y => y / nrm);
}


// ------------------------------------------------------------------ GLM: encoder + any number of loaded packs
export class GLM {
  constructor(encoder) { this.enc = encoder; this.packs = new Map(); }
  load(pack) { this.packs.set(pack.id, pack); }
  unload(id) { return this.packs.delete(id); }

  async ask(question, opts = {}) {
    const T0 = now(), tm = {};
    if (!this.packs.size) return { ok: false, reason: "No knowledge pack is loaded. Load one to ask questions.", timings: tm };
    if (opts.pack && !this.packs.has(opts.pack)) return { ok: false, reason: "That knowledge pack is not loaded. Load it to ask questions.", timings: tm };
    const sel = opts.pack ? [this.packs.get(opts.pack)] : [...this.packs.values()].filter(p => p.kind !== "wikidata");
    if (sel.length === 1 && sel[0].kind === "wikidata") {   // general pack (assets/glm/wiki.js): its own linker and scorer
      const r = await sel[0].ask(question, this.enc, opts);
      return { pack: sel[0].id, ...r };
    }
    if (!sel.length) return { ok: false, reason: "No movie pack is loaded.", timings: tm };
    let t0 = now(), link = null, pack = null;
    for (const p of sel) {
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
    const out = { ...r, pack: pack.id, link, masked, timings: tm };
    if (r.ok && opts.viz) {   // drawing data, made after the timed part
      const t1 = now();
      out.viz = pack.subgraph(link.id, [{ e: r.path_e, kind: "chosen" }, ...r.alternatives.map(a => ({ e: a.path_e, kind: "alt" }))]);
      out.viz_ms = now() - t1;
    }
    return out;
  }
}
