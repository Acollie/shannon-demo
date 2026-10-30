// GLM v1 general pack ("shannon-pack/2", kind "wikidata-sharded"): a Wikidata core loaded lazily in shards, answered by
// the KG16 R0 model (frozen MiniLM + KG15 S1-sym path scorer + H1 answer-type head + direction head, ~0.1M trained) with
// the KG17-style fast linker (exact alias index + label-prefix index, KG15 linker MLP). Pure module (no DOM): the page
// and the Node parity check (tools/glm_parity/parity_general.mjs) run the same code.
// Port of shannon/g/kg15 (paths.py walk / question / arg_table, scorer.py OpenComposer s1 plain, train.py predict) and
// shannon/g/kg17/link17.py features_fast. The Python reference for parity is tools/glm_general/ref.py.
//
// Candidate listing does not walk 2-hop sets: it needs only the topic's adjacency and, for every neighbour V, V's
// relation signature (edge type, count, self loop). npaths(r1, r2) = sum over V of |N_r2(V) minus {V, topic}|, the same
// numbers kg15.paths.walk counts. Only the top candidate (and the drawn graph) is materialised.

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const NR = 4096;                       // > number of signed relation ids (2 x 1,089)
const D = 16, DD = 256;
export const OPS = ["set", "count", "yesno", "argmax", "argmin", "compare"];

// ------------------------------------------------------------------ text: KG15 common.norm
export function knorm(s) {
  return String(s).normalize("NFKD").replace(/[^\x00-\x7f]/g, "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).join(" ");
}
function fnv(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
function ngrams(toks, nmax) {
  const out = [];
  for (let n = 1; n <= nmax; n++) for (let i = 0; i + n <= toks.length; i++) out.push([i, n, toks.slice(i, i + n).join(" ")]);
  return out;
}
// float32 -> nearest float16 -> float32 (round half to even), like numpy's float16 cast
function f16r(x) {
  if (!Number.isFinite(x) || x === 0) return x;
  const sg = x < 0 ? -1 : 1, a = Math.abs(x);
  if (a >= 65520) return sg * Infinity;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  if (2 ** (e + 1) <= a) e++;
  if (e < -14) e = -14;
  const ulp = 2 ** (e - 10), q = a / ulp, f = Math.floor(q), d = q - f;
  const r = d > 0.5 || (d === 0.5 && (f & 1)) ? f + 1 : f;
  return sg * r * ulp;
}
// erf (W. J. Cody's rational approximations, |rel err| < 1e-15 over the real line): exact GELU like torch.nn.GELU()
function erf(x) {
  const ax = Math.abs(x);
  let r;
  if (ax < 0.5) {
    const t = x * x;
    const p = [3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1];
    const q = [2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03];
    let n = p[4] * t, d = t;
    for (let i = 0; i < 3; i++) { n = (n + p[i]) * t; d = (d + q[i]) * t; }
    return x * (n + p[3]) / (d + q[3]);
  }
  if (ax < 4) {
    const p = [5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8];
    const q = [1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03];
    let n = p[8] * ax, d = ax;
    for (let i = 0; i < 7; i++) { n = (n + p[i]) * ax; d = (d + q[i]) * ax; }
    r = (n + p[7]) / (d + q[7]);
  } else {
    const p = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
    const q = [2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
    const z = 1 / (ax * ax);
    let n = p[5] * z, d = z;
    for (let i = 0; i < 4; i++) { n = (n + p[i]) * z; d = (d + q[i]) * z; }
    r = z * (n + p[4]) / (d + q[4]);
    r = (0.56418958354775628695 - r) / ax;
  }
  const xs = Math.trunc(ax * 16) / 16, del = (ax - xs) * (ax + xs);
  r = Math.exp(-xs * xs) * Math.exp(-del) * r;
  return x >= 0 ? 1 - r : r - 1;
}
const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

// ------------------------------------------------------------------ shards
function varints(b, p, n, out) {        // read n LEB128 values from b at p into out; returns the new position
  for (let k = 0; k < n; k++) {
    let x = 0, mul = 1, c;
    do { c = b[p++]; x += (c & 0x7f) * mul; mul *= 128; } while (c & 0x80);
    out[k] = x;
  }
  return p;
}
function rv(st) { const b = st.b; let x = 0, mul = 1, c; do { c = b[st.p++]; x += (c & 0x7f) * mul; mul *= 128; } while (c & 0x80); return x; }

class Shard {      // u32 lo | u32 n | u32 off[n+1] | body
  constructor(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.lo = dv.getUint32(0, true); this.n = dv.getUint32(4, true);
    this.off = new Uint32Array(this.n + 1);
    for (let i = 0; i <= this.n; i++) this.off[i] = dv.getUint32(8 + 4 * i, true);
    this.body = bytes.subarray(8 + 4 * (this.n + 1));
  }
  cursor(v) { return { b: this.body, p: this.off[v - this.lo] }; }
}
function upper(lo, v) { let a = 0, b = lo.length; while (a < b) { const m = (a + b) >> 1; if (lo[m] <= v) a = m + 1; else b = m; } return a - 1; }

// ------------------------------------------------------------------ the pack
export class WikiPack {
  // manifest: parsed manifest.json; get(path) -> Promise<Uint8Array> (the page: Cache Storage + fetch; Node: disk)
  constructor(manifest, get) {
    this.m = manifest; this.id = manifest.id; this.kind = "wikidata"; this.get = get;
    this.F = manifest.files; this.sc = manifest.scorer;
    this.edgeNames = []; this.edgeLabels = []; this.relId = [];
    for (const r of manifest.relations) {
      this.edgeNames.push(r.id, r.id + manifest.inverse_suffix);
      this.edgeLabels.push(r.label, r.inverse_label);
    }
    this.NE = manifest.counts.entities; this.n = manifest.counts.nodes;
    this.files = new Map();             // path -> Promise<Uint8Array|Shard|...>
    this.adjC = new Map(); this.sigC = new Map(); this.litC = new Map(); this.nameC = new Map(); this.idxC = new Map();
    this.stats = { files: 0, bytes: 0 };
    this.inflight = 0; this.queue = [];
    this.maxInflight = 24;
  }

  async init() {
    const t0 = now();
    const [w, nb] = await Promise.all([this.raw(this.F.weights.path), this.raw(this.F.node.path)]);
    const W = new Float32Array(w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength)), L = this.F.weights.layout;
    const get = k => W.subarray(L[k].offset, L[k].offset + L[k].shape.reduce((a, b) => a * b, 1));
    for (const k of Object.keys(L)) this[k] = get(k);
    this.tau = Math.exp(this.logtau[0]); this.betaV = this.beta[0];
    const N = this.n;
    this.deg = new Float64Array(N); this.pdeg = new Float64Array(N);
    let p = varints(nb, 0, N, this.deg);
    p = varints(nb, p, N, this.pdeg);
    this.lkind = nb.subarray(p, p + (N - this.NE));     // 1 = quantity, 0 = date
    this.Op = new Map();                                 // signed relation id -> I + L (or I + L^T), Float64Array(256)
    this.buildMs = now() - t0;
    return this;
  }

  // ---- fetching (at most maxInflight requests at once)
  async limited(fn) {
    if (this.inflight >= this.maxInflight) await new Promise(r => this.queue.push(r));
    this.inflight++;
    try { return await fn(); } finally { this.inflight--; const nx = this.queue.shift(); if (nx) nx(); }
  }
  raw(path) {
    let p = this.files.get(path);
    if (!p) {
      p = this.limited(() => this.get(path)).then(b => { this.stats.files++; this.stats.bytes += b.length; return b; });
      this.files.set(path, p);
      p.catch(() => this.files.delete(path));
    }
    return p;
  }
  async shards(kind, ids, cache) {
    const lo = this.F[kind].lo, want = new Set();
    for (const v of ids) { const s = upper(lo, v); if (!cache.has(s)) want.add(s); }
    await Promise.all([...want].map(async s => {
      const b = await this.raw(`${this.F[kind].dir}/${String(s).padStart(3, "0")}.bin`);
      if (!cache.has(s)) cache.set(s, new Shard(b));
    }));
  }
  shardOf(kind, cache, v) { return cache.get(upper(this.F[kind].lo, v)); }
  needAdj(ids) { return this.shards("adj", ids, this.adjC); }
  needSig(ids) { return this.shards("sig", ids, this.sigC); }
  needLit(ids) { return this.shards("lit", ids.filter(v => v < this.NE), this.litC); }
  async needNames(ids) {
    const per = this.F.names.per_shard, want = new Set();
    for (const v of ids) { const s = Math.floor(v / per); if (!this.nameC.has(s)) want.add(s); }
    await Promise.all([...want].map(async s => {
      const t = new TextDecoder().decode(await this.raw(`${this.F.names.dir}/${String(s).padStart(3, "0")}.txt`));
      const l = t.split("\n"); if (l[l.length - 1] === "") l.pop();
      this.nameC.set(s, l);
    }));
  }
  nm(v) { const per = this.F.names.per_shard, l = this.nameC.get(Math.floor(v / per)); return l ? l[v % per] : `#${v}`; }
  async needLitVals() {
    if (!this.litval) { const b = await this.raw(this.F.litval.path); this.litval = new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
  }

  // adjacency of v: {srel: [...], st: [...], mem: Int32Array} (groups sorted by srel; members ascending)
  adj(v) {
    let a = this.adjC.get("n" + v);
    if (a) return a;
    const sh = this.shardOf("adj", this.adjC, v), st = sh.cursor(v);
    const ng = rv(st), srel = new Int32Array(ng), start = new Int32Array(ng + 1);
    const mem = [];
    let s = 0;
    for (let j = 0; j < ng; j++) {
      s += rv(st); srel[j] = s; const c = rv(st); start[j] = mem.length;
      let x = 0;
      for (let k = 0; k < c; k++) { x = k === 0 ? rv(st) : x + rv(st); mem.push(x); }
    }
    start[ng] = mem.length;
    a = { srel, start, mem: Int32Array.from(mem) };
    if (this.adjC.size < 200000) this.adjC.set("n" + v, a);
    return a;
  }
  group(v, e) {     // members of v along signed relation e (Int32Array view) or null
    const a = this.adj(v);
    let lo = 0, hi = a.srel.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (a.srel[m] < e) lo = m + 1; else hi = m; }
    return lo < a.srel.length && a.srel[lo] === e ? a.mem.subarray(a.start[lo], a.start[lo + 1]) : null;
  }
  sig(v) {          // [srel, cnt, self] per group
    const sh = this.shardOf("sig", this.sigC, v), st = sh.cursor(v);
    const ng = rv(st), out = new Int32Array(3 * ng);
    let s = 0;
    for (let j = 0; j < ng; j++) { s += rv(st); const x = rv(st); out[3 * j] = s; out[3 * j + 1] = Math.floor(x / 2); out[3 * j + 2] = x & 1; }
    return out;
  }
  lits(v) {         // forward literal facts of entity v: [signed relation (even), literal node] pairs
    if (v >= this.NE) return new Int32Array(0);
    const sh = this.shardOf("lit", this.litC, v), st = sh.cursor(v);
    const n = rv(st), out = new Int32Array(2 * n);
    for (let k = 0; k < n; k++) { out[2 * k] = 2 * rv(st); out[2 * k + 1] = this.NE + rv(st); }
    return out;
  }

  // ---- name index (exact aliases + label prefixes), sharded by first word
  idxShard(k) {
    const t = k.split(" "), split = this.F.idx.split_first_words;
    if (!this._split) this._split = new Set(split);
    const key = this._split.has(t[0]) && t.length > 1 ? t[0] + " " + t[1] : t[0];
    return fnv(key) % this.F.idx.shards;
  }
  async needIdx(keys) {
    const want = new Set();
    for (const k of keys) { const s = this.idxShard(k); if (!this.idxC.has(s)) want.add(s); }
    await Promise.all([...want].map(async s => {
      const t = new TextDecoder().decode(await this.raw(`${this.F.idx.dir}/${String(s).padStart(4, "0")}.txt`));
      const m = new Map();
      for (const line of t.split("\n")) { if (!line) continue; const a = line.split("\t"); m.set(a[0], [a[1], a[2]]); }
      this.idxC.set(s, m);
    }));
  }
  exact(k) { const r = this.idxC.get(this.idxShard(k)).get(k); return r && r[0] ? r[0].split(",").map(x => parseInt(x, 36)) : null; }
  prefix(k) {
    const r = this.idxC.get(this.idxShard(k)).get(k);
    if (!r || !r[1]) return null;
    return r[1].split(",").map(x => { const [a, b] = x.split(":"); return [parseInt(a, 36), +b]; });
  }

  // ---- linker (kg17.link17.features_fast + kg15.link.Linker)
  async link(question, enc) {
    const ws = knorm(question), toks = ws ? ws.split(" ") : [];
    const lk = this.m.linker, grams = ngrams(toks, lk.maxn);
    await this.needIdx(grams.map(g => g[2]));
    const best = new Map(), L = Math.max(toks.length, 1);
    const hits = grams.filter(g => this.exact(g[2]));
    const spans = hits.map(([i, n]) => [i, i + n]);
    const exactIds = new Set();
    for (const [, , s] of hits) for (const c of this.exact(s)) exactIds.add(c);
    await this.needNames([...exactIds]);
    for (const [i, n, s] of hits) {
      const maximal = !spans.some(([a, b]) => a <= i && i + n <= b && (b - a) > n);
      const ids = this.exact(s);
      ids.forEach((c, rk) => {
        const r = [1, 1, n, s.length, maximal ? 1 : 0, ids.length, 1 / (1 + rk), knorm(this.nm(c)) === s ? 1 : 0, i / L, i, n];
        const o = best.get(c);    // python: if c not in best or r[:4] > best[c][:4] (here r[0] = r[1] = 1 always)
        if (!o || r[2] > o[2] || (r[2] === o[2] && r[3] > o[3])) best.set(c, r);
      });
    }
    const ph = grams.filter(g => this.prefix(g[2])).map(([i, n, s]) => [n, i, s]).sort((x, y) => y[0] - x[0] || y[1] - x[1]);
    let added = 0;
    for (const [n, i, s] of ph) {
      const ids = this.prefix(s);
      for (let rk = 0; rk < ids.length; rk++) {
        const [c, lt] = ids[rk];
        if (best.has(c)) continue;
        best.set(c, [0, n / Math.max(lt, 1), n, s.length, 1, ids.length, 1 / (1 + rk), 0, i / L, i, n]);
        if (++added >= lk.prefix_new) break;
      }
      if (added >= lk.prefix_new) break;
    }
    const nodes = [...best.keys()].sort((x, y) => x - y);
    if (!nodes.length) return { topics: [], tlog: [], nodes, toks };
    await this.needNames(nodes);
    const t0 = now();
    const vecs = await enc.sent([question, ...nodes.map(c => this.nm(c))]);
    const encMs = now() - t0;
    const qv = vecs[0], lg = [];
    for (let k = 0; k < nodes.length; k++) {
      const c = nodes[k], [ex, cs, n, ln, mx, amb, rk, isl, p] = best.get(c);
      const nv = vecs[k + 1];
      let ctx = 0;
      for (let i = 0; i < 384; i++) ctx += qv[i] * nv[i];
      ctx = Math.fround(ctx);
      const X = [ex, cs, n / 5, ln / 30, Math.log1p(this.deg[c]) / 10, mx, Math.log1p(amb) / 5, rk, isl, p, ctx, ctx * ex, cs * (1 - ex), 1].map(Math.fround);
      let out = this.link_b2[0];
      for (let h = 0; h < 32; h++) {
        let z = this.link_b1[h];
        for (let f = 0; f < 14; f++) z += this.link_W1[h * 14 + f] * X[f];
        out += this.link_W2[h] * gelu(z);
      }
      lg.push(out);
    }
    const o = nodes.map((_, k) => k).sort((x, y) => lg[y] - lg[x]);
    const top = o.slice(0, lk.top_topics);
    return { topics: top.map(k => nodes[k]), tlog: top.map(k => lg[k]), spans: top.map(k => best.get(nodes[k]).slice(9, 11)),
      nodes, toks, qsent: qv, enc_ms: encMs, n_link_cands: nodes.length };
  }

  // ---- candidates (kg15.paths.question, without materialising 2-hop sets)
  async candidates(topics) {
    const rows = [], H2 = this.sc.hop2_max;
    await this.needAdj(topics);
    const per = [];
    for (let si = 0; si < topics.length; si++) {
      const t = topics[si], a = this.adj(t), nb = [];
      let tot = 0;
      for (let j = 0; j < a.srel.length; j++) {
        let c = 0;
        for (let k = a.start[j]; k < a.start[j + 1]; k++) { const v = a.mem[k]; if (v === t) continue; c++; tot += this.pdeg[v]; nb.push(v); }
        if (c) rows.push([si, a.srel[j], -1, c]);
      }
      per.push({ t, a, nb, tot });
    }
    const need = [];
    for (const p of per) if (p.tot > 0 && p.tot <= H2) for (const v of p.nb) need.push(v);
    const tf = now();
    await this.needSig(need);
    const fetchMs = now() - tf;
    for (let si = 0; si < per.length; si++) {
      const { t, a, tot } = per[si];
      if (!(tot > 0 && tot <= H2)) continue;
      const back = new Set();                // (V, e): t has an edge e to V  <=>  V has an edge e^1 to t
      for (let j = 0; j < a.srel.length; j++) for (let k = a.start[j]; k < a.start[j + 1]; k++) back.add(a.mem[k] * NR + (a.srel[j] ^ 1));
      const cnt = new Map();
      for (let j = 0; j < a.srel.length; j++) {
        const r1 = a.srel[j];
        for (let k = a.start[j]; k < a.start[j + 1]; k++) {
          const v = a.mem[k];
          if (v === t) continue;
          const sg = this.sig(v);
          for (let q = 0; q < sg.length; q += 3) {
            const r2 = sg[q], surv = sg[q + 1] - sg[q + 2] - (back.has(v * NR + r2) ? 1 : 0);
            if (surv > 0) { const key = r1 * NR + r2; cnt.set(key, (cnt.get(key) || 0) + surv); }
          }
        }
      }
      for (const [key, c] of cnt) rows.push([si, Math.floor(key / NR), key % NR, c]);
    }
    rows.sort((x, y) => y[3] - x[3] || x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);
    return { rows: rows.slice(0, this.sc.cand_cap), n_full: rows.length, fetch_ms: fetchMs };
  }

  // reached set of (start, r1, r2): Map node -> number of paths (kg15.paths.walk / realise)
  async realise(t, r1, r2, keepParents = false) {
    await this.needAdj([t]);
    const g1 = this.group(t, r1) || new Int32Array(0), out = new Map(), parents = keepParents ? new Map() : null;
    if (r2 < 0) { for (const v of g1) if (v !== t) out.set(v, 1); return { nd: out, parents }; }
    const V = [...g1].filter(v => v !== t);
    await this.needAdj(V);
    for (const v of V) {
      const g2 = this.group(v, r2);
      if (!g2) continue;
      for (const w of g2) {
        if (w === v || w === t) continue;
        out.set(w, (out.get(w) || 0) + 1);
        if (parents) { let l = parents.get(w); if (!l) parents.set(w, (l = [])); if (l.length < 4) l.push(v); }
      }
    }
    return { nd: out, parents };
  }
  setTop(nd) {   // most paths, ties: highest degree, then smallest id
    let best = -1, bc = -1, bd = -1;
    for (const [v, c] of nd) if (c > bc || (c === bc && (this.deg[v] > bd || (this.deg[v] === bd && v < best)))) { best = v; bc = c; bd = this.deg[v]; }
    return best;
  }
  // kg15.paths.arg_table: per attribute held by >= 2 members, the max / min member (ties: highest degree, then member order)
  async argTable(nodes) {
    await this.needLit(nodes); await this.needLitVals();
    const per = new Map();      // attr -> [[member index, value]]
    nodes.forEach((v, mi) => {
      const l = this.lits(v), mx = new Map();
      for (let k = 0; k < l.length; k += 2) { const x = this.litval[l[k + 1] - this.NE]; const o = mx.get(l[k]); if (o === undefined || x > o) mx.set(l[k], x); }
      for (const [r, x] of mx) { let a = per.get(r); if (!a) per.set(r, (a = [])); a.push([mi, x]); }
    });
    const at = [], amx = [], amn = [], vmx = [], vmn = [];
    for (const r of [...per.keys()].sort((x, y) => x - y)) {
      const a = per.get(r);
      if (a.length < 2) continue;
      a.sort((x, y) => x[0] - y[0]);
      let hi = -Infinity, lo = Infinity;
      for (const [, x] of a) { if (x > hi) hi = x; if (x < lo) lo = x; }
      const pick = val => { let b = -1, bd = -1; for (const [mi, x] of a) if (x === val && this.deg[nodes[mi]] > bd) { b = nodes[mi]; bd = this.deg[nodes[mi]]; } return b; };
      at.push(r); amx.push(pick(hi)); amn.push(pick(lo)); vmx.push(hi); vmn.push(lo);
    }
    return { at, mx: amx, mn: amn, vmx, vmn };
  }

  // ---- scorer (OpenComposer s1, rel='sym', prod='plain'; relation operators precomputed in the pack)
  qvec(T, n) {
    const d = 384, mu = this.mu, sd = this.sd, Q = new Float64Array(DD), x = new Float64Array(d);
    let asum = 0;
    for (let t = 0; t < n; t++) {
      for (let i = 0; i < d; i++) x[i] = (T[t * d + i] - mu[i]) / sd[i];
      let z = this.ba[0];
      for (let i = 0; i < d; i++) z += this.wa[i] * x[i];
      const a = 1 / (1 + Math.exp(-z));
      asum += a;
      for (let k = 0; k < DD; k++) {
        let s = this.bq[k];
        const w = this.Wq, o = k * d;
        for (let i = 0; i < d; i++) s += w[o + i] * x[i];
        Q[k] += a * s;
      }
    }
    const den = Math.max(asum, 1e-6);
    let nrm = 0;
    for (let k = 0; k < DD; k++) { Q[k] /= den; nrm += Q[k] * Q[k]; }
    nrm = Math.max(Math.sqrt(nrm), 1e-6);
    const sc = Math.min(2 * Math.sqrt(D) / nrm, 1);
    for (let k = 0; k < DD; k++) Q[k] *= sc;
    for (let i = 0; i < D; i++) Q[i * D + i] += 1;
    return normed(Q);
  }
  op(e) {
    let O = this.Op.get(e);
    if (!O) {
      O = new Float64Array(DD);
      const L = this.L, o = (e >> 1) * DD, tr = e & 1;
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) O[i * D + j] = tr ? L[o + j * D + i] : L[o + i * D + j];
      for (let i = 0; i < D; i++) O[i * D + i] += 1;
      this.Op.set(e, O);
    }
    return O;
  }
  seqScore(q, a, b) {
    let P = this.op(a);
    if (b >= 0) P = matmul(P, this.op(b));
    let s = 0, n = 0;
    for (let k = 0; k < DD; k++) { s += q[k] * P[k]; n += P[k] * P[k]; }
    return this.tau * s / Math.max(Math.sqrt(n), 1e-12);
  }
  head(W, b, S, k) { let z = b[k]; for (let i = 0; i < 384; i++) z += W[k * 384 + i] * S[i]; return z; }

  // ---- one question -> answer (kg15.train.predict) + explanation
  async ask(question, enc, opts = {}) {
    const T0 = now(), tm = {}, f0 = { ...this.stats };
    let t0 = now();
    const link = await this.link(question, enc);
    tm.link_ms = now() - t0 - (link.enc_ms || 0);
    if (!link.topics.length) return { ok: false, reason: "I could not find a name from the general pack in the question.", timings: tm };
    await this.needNames(link.topics);
    t0 = now();
    const { tok, n } = await enc.tokens(question, this.sc.max_tokens);
    const T = new Float32Array(tok.length);
    for (let i = 0; i < tok.length; i++) T[i] = this.sc.tok_fp16 ? f16r(tok[i]) : tok[i];
    const S = link.qsent;
    tm.encoder_ms = now() - t0 + (link.enc_ms || 0);
    t0 = now();
    const q = this.qvec(T, n);
    let op = 0, best = -Infinity;
    for (let k = 0; k < 6; k++) { const z = this.head(this.h1_W, this.h1_b, S, k); if (z > best) { best = z; op = k; } }
    const dirz = this.head(this.dir_W, this.dir_b, S, 0);
    tm.tagger_ms = now() - t0;
    t0 = now();
    const cand = await this.candidates(link.topics);
    tm.fetch_ms = cand.fetch_ms;
    tm.paths_ms = now() - t0 - cand.fetch_ms;
    t0 = now();
    const tl = [...link.tlog, 0, 0, 0, 0].slice(0, 4);
    const scored = cand.rows.map(([si, a, b, np], i) => ({ si, a, b, np, i, score: this.seqScore(q, a, b) + this.betaV * tl[si] }));
    const order = scored.slice().sort((x, y) => y.score - x.score || x.i - y.i);
    tm.score_ms = now() - t0;
    t0 = now();
    const top = order[0] || null, topics = link.topics;
    let ans = null, node = null, how = OPS[op], chosenSet = null, attr = null;
    const ents = topics.filter(x => x < this.NE).slice(0, 2);
    if (top) chosenSet = await this.realise(topics[top.si], top.a, top.b, true);
    const scoreAttrs = at => { let j = 0, bs = -Infinity; at.forEach((r, k) => { const s = this.seqScore(q, r, -1); if (s > bs) { bs = s; j = k; } }); return j; };
    const size = chosenSet ? chosenSet.nd.size : 0, setTop = chosenSet ? this.setTop(chosenSet.nd) : -1;
    if (op === 5) {
      const ct = ents.length >= 2 ? await this.argTable(ents) : { at: [] };
      if (ct.at.length) {
        const j = scoreAttrs(ct.at), big = dirz < 0;
        node = big ? ct.mx[j] : ct.mn[j];
        attr = { e: ct.at[j], larger: big, values: ents.map(v => this.valueOf(v, ct.at[j])), ents };
      } else if (top) { node = setTop; how = "set(fallback)"; }
    } else if (op === 2) {
      ans = top ? topics.some(x => x !== topics[top.si] && chosenSet.nd.has(x)) : false;
    } else if (top) {
      if (op === 1) ans = size;
      else if (op === 0) node = setTop;
      else {
        node = setTop; how = "set(fallback)";
        if (size >= 2 && size <= this.sc.arg_max_set) {
          const nd = [...chosenSet.nd.keys()].sort((x, y) => x - y);
          const t = await this.argTable(nd);
          if (t.at.length) {
            const j = scoreAttrs(t.at);
            node = op === 3 ? t.mx[j] : t.mn[j]; how = OPS[op];
            attr = { e: t.at[j], larger: op === 3, value: op === 3 ? t.vmx[j] : t.vmn[j], held: null };
          }
        }
      }
    }
    if (node !== null && node >= 0) ans = this.answerOf(node);
    tm.walk_ms = now() - t0;
    tm.total_ms = now() - T0;
    tm.download_bytes = this.stats.bytes - f0.bytes; tm.download_files = this.stats.files - f0.files;
    const core = { op: OPS[op], how, answer: ans, node, topics, tlog: link.tlog, top: top ? [top.si, [top.a, top.b]] : null,
      top3: order.slice(0, 3).map(c => [c.si, [c.a, c.b]]), n_cands: cand.rows.length, n_full: cand.n_full };
    if (opts.core) return { ok: !!top || ans !== null, ...core, timings: tm };
    return this.explain(question, link, core, order, chosenSet, attr, tm, opts);
  }

  answerOf(v) { return { node: v, kind: v < this.NE ? "entity" : this.lkind[v - this.NE] ? "quantity" : "date" }; }
  valueOf(v, e) { const l = this.lits(v); let b = null; for (let k = 0; k < l.length; k += 2) if (l[k] === e) { const x = this.litval[l[k + 1] - this.NE]; if (b === null || x > b[0]) b = [x, l[k + 1]]; } return b ? b[1] : null; }

  // ---- result for the page: path, hops, alternatives, drawing data
  async explain(question, link, core, order, chosen, attr, tm, opts) {
    const t0 = now();
    const top = order[0], topics = link.topics;
    const show = new Set(topics);
    let hops = [], answers = [], others = [], nFront = 0, path = [], path_e = [], topicId = topics[0];
    if (top) {
      topicId = topics[top.si];
      path_e = top.b >= 0 ? [top.a, top.b] : [top.a];
      path = path_e.map(e => this.edgeNames[e]);
      const g1 = [...(this.group(topicId, top.a) || [])].filter(v => v !== topicId);
      const byDeg = l => l.slice().sort((x, y) => this.deg[y] - this.deg[x] || x - y);
      const ranked = [...chosen.nd].sort((x, y) => y[1] - x[1] || this.deg[y[0]] - this.deg[x[0]] || x[0] - y[0]);
      nFront = ranked.length;
      const s1 = top.b >= 0 ? byDeg(g1).slice(0, 4) : ranked.slice(0, 4).map(x => x[0]);
      s1.forEach(v => show.add(v));
      hops.push({ edge: this.edgeNames[top.a], label: this.edgeLabels[top.a], n: g1.length, sample: s1 });
      if (top.b >= 0) { const s2 = ranked.slice(0, 4).map(x => x[0]); s2.forEach(v => show.add(v)); hops.push({ edge: this.edgeNames[top.b], label: this.edgeLabels[top.b], n: ranked.length, sample: s2 }); }
      ranked.slice(0, 12).forEach(x => show.add(x[0]));
      others = ranked.slice(0, 12);
    }
    if (core.node !== null && core.node >= 0) show.add(core.node);
    // alternatives: the next candidates (1-step ones answered from the topic's own facts)
    const alts = [];
    for (const c of order.slice(1, 4)) {
      const t = topics[c.si], pe = c.b >= 0 ? [c.a, c.b] : [c.a];
      let ans = [], n = c.np;
      if (c.b < 0) { const g = [...(this.group(t, c.a) || [])].filter(v => v !== t); n = g.length; ans = g.sort((x, y) => this.deg[y] - this.deg[x] || x - y).slice(0, 3); ans.forEach(v => show.add(v)); }
      alts.push({ t, path_e: pe, score: c.score, ans, n, two: c.b >= 0 });
    }
    const viz = top && opts.viz ? this.subgraph(topicId, top, chosen, alts, show) : null;
    if (attr) {
      await this.needLitVals();
      if (attr.values) attr.values.forEach(v => v !== null && show.add(v));
      else if (core.node !== null && core.node >= 0) { attr.vnode = this.valueOf(core.node, attr.e); if (attr.vnode !== null) show.add(attr.vnode); }
    }
    await this.needNames([...show]);
    const N = v => this.nm(v);
    if (top) for (const h of hops) h.sample = h.sample.map(N);
    // the answer in words
    let answerText = [], explainOp = "";
    const op = core.op;
    if (op === "count") { answerText = [String(core.answer)]; explainOp = `count of the ${nFront.toLocaleString()} ${nFront === 1 ? "thing" : "things"} reached`; }
    else if (op === "yesno") {
      answerText = [core.answer ? "Yes" : "No"];
      const oth = topics.filter(x => !top || x !== topics[top.si]);
      explainOp = oth.length ? `yes/no: is ${oth.map(N).join(" or ")} among the ${nFront.toLocaleString()} reached?` : "yes/no: no second name to look for";
    } else if (core.node !== null && core.node >= 0) {
      answerText = [N(core.node)];
      if (attr && op === "compare") explainOp = `compare ${attr.ents.map((v, k) => `${N(v)} (${attr.values[k] === null ? "?" : N(attr.values[k])})`).join(" and ")} on ${this.edgeLabels[attr.e]}: the ${attr.larger ? "larger" : "smaller"} wins`;
      else if (attr) explainOp = `${attr.larger ? "largest" : "smallest"} ${this.edgeLabels[attr.e]} of the ${nFront.toLocaleString()} reached${attr.vnode != null ? ` (${N(attr.vnode)})` : ""}`;
      else explainOp = core.how === "set(fallback)" ? `${op === "set" ? "the" : `${op} found no attribute; the`} node reached by the most paths (ties: the best-known)` : "the node reached by the most paths (ties: the best-known)";
    } else explainOp = "no answer";
    if (viz) for (const nd of viz.nodes) nd.name = N(nd.id);
    if (attr && attr.values) attr.valueText = attr.values.map(v => (v === null ? "?" : N(v)));
    const alternatives = alts.map(a => ({ path: a.path_e.map(e => this.edgeNames[e]), labels: a.path_e.map(e => this.edgeLabels[e]), score: a.score,
      answers: a.two ? [] : a.ans.map(N), n: a.n, path_e: a.path_e, two: a.two, from: N(a.t) }));
    tm.explain_ms = now() - t0;
    const span = this.spanOf(question, link, top ? top.si : 0);
    return {
      ok: !!top || core.answer !== null, pack: this.id, kind: "wikidata", ...core,
      topic: N(topicId), topic_id: topicId, topics_named: topics.map(N), path, path_e, labels: path_e.map(e => this.edgeLabels[e]),
      score: top ? top.score : 0, walked_rank: 0, answers: answerText, explain_op: explainOp,
      others: others.filter(([v]) => v !== core.node).slice(0, 10).map(([v, c]) => [N(v), c]), n_frontier: nFront,
      hops, alternatives, top5: order.slice(0, 5).map(c => [[c.a, c.b].filter(x => x >= 0).map(e => this.edgeNames[e]), c.score]),
      attr: attr && { label: this.edgeLabels[attr.e], larger: attr.larger, values: attr.valueText || null, ents: attr.ents ? attr.ents.map(N) : null, value: attr.vnode != null ? N(attr.vnode) : null },
      link: span, viz, timings: tm,
    };
  }

  // character span of the chosen topic's matched words in the question (for highlighting), or null
  spanOf(question, link, si) {
    const sp = link.spans && link.spans[si];
    if (!sp) return { name: this.nm(link.topics[si]), id: link.topics[si] };
    const [i, n] = sp, words = [];
    for (const m of question.matchAll(/[^\s]+/g)) {
      for (const w of knorm(m[0]).split(" ").filter(Boolean)) words.push([w, m.index, m.index + m[0].length]);
    }
    const r = { name: this.nm(link.topics[si]), id: link.topics[si] };
    if (words.length === link.toks.length && n > 0) {
      let a = words[i][1], b = words[i + n - 1][2];
      while (b > a && /[^\p{L}\p{N}]/u.test(question[b - 1])) b--;
      while (a < b && /[^\p{L}\p{N}]/u.test(question[a])) a++;
      Object.assign(r, { a, b, span: question.slice(a, b) });
    }
    return r;
  }

  // local graph for the thinking-graph canvas (same shape as Pack.subgraph in glm.js)
  subgraph(topic, top, chosen, alts, show) {
    const nodes = new Map(), edges = [], keys = new Set(), stats = [];
    const add = (id, depth, kind) => { let n = nodes.get(id); if (!n) { if (nodes.size >= 150) return null; n = { id, name: "", depth, kind, isNew: false, answer: false, taught: false }; nodes.set(id, n); } return n; };
    const edge = (a, b, e, kind, hop, pi) => { const k = a + ">" + e + ">" + b; if (keys.has(k) || !nodes.has(a) || !nodes.has(b)) return; keys.add(k); edges.push({ a, b, e, label: this.edgeLabels[e], kind, hop, pi, taught: false }); };
    add(topic, 0, "chosen").topic = true;
    const ranked = [...chosen.nd].sort((x, y) => y[1] - x[1] || this.deg[y[0]] - this.deg[x[0]] || x[0] - y[0]);
    const topC = ranked.length ? ranked[0][1] : 0;
    const L = 12;
    if (top.b < 0) {
      for (const [v, c] of ranked.slice(0, L)) { const n = add(v, 1, "chosen"); if (n && c === topC) n.answer = true; edge(topic, v, top.a, "chosen", 0, 0); }
      stats.push({ hop: 1, label: this.edgeLabels[top.a], n: ranked.length, shown: Math.min(L, ranked.length) });
    } else {
      const mids = new Set(), last = ranked.slice(0, L);
      for (const [w] of last) for (const v of (chosen.parents.get(w) || []).slice(0, 2)) { if (mids.size >= L && !mids.has(v)) continue; mids.add(v); }
      for (const v of mids) { add(v, 1, "chosen"); edge(topic, v, top.a, "chosen", 0, 0); }
      for (const [w, c] of last) {
        const ps = (chosen.parents.get(w) || []).filter(v => mids.has(v));
        if (!ps.length) continue;
        const n = add(w, 2, "chosen"); if (n && c === topC) n.answer = true;
        for (const v of ps.slice(0, 2)) edge(v, w, top.b, "chosen", 1, 0);
      }
      const g1 = (this.group(topic, top.a) || []).length;
      stats.push({ hop: 1, label: this.edgeLabels[top.a], n: g1, shown: mids.size }, { hop: 2, label: this.edgeLabels[top.b], n: ranked.length, shown: last.filter(([w]) => nodes.has(w)).length });
    }
    alts.forEach((a, k) => {
      if (a.two || a.t !== topic) return;
      for (const v of a.ans) { add(v, 1, "alt"); edge(topic, v, a.path_e[0], "alt", 0, k + 1); }
    });
    const firsts = new Set([top.a, ...alts.map(a => a.path_e[0])]);
    const ad = this.adj(topic);
    let nc = 0;
    for (let j = 0; j < ad.srel.length && nc < 24; j++) {
      const e = ad.srel[j];
      if (firsts.has(e)) continue;
      const ms = [...ad.mem.subarray(ad.start[j], ad.start[j + 1])].filter(v => v !== topic).sort((x, y) => this.deg[y] - this.deg[x] || x - y).slice(0, 2);
      for (const v of ms) { if (nc >= 24) break; if (!nodes.has(v)) nc++; add(v, 1, "context"); edge(topic, v, e, "context", 0, -1); }
    }
    for (const id of nodes.keys()) show.add(id);
    const reached = 1 + (this.group(topic, top.a) || []).length + (top.b >= 0 ? ranked.length : 0);
    const seen = Math.max(reached, nodes.size);
    return { nodes: [...nodes.values()], edges, stats, reached: seen, hidden: Math.max(0, seen - nodes.size), taught_answers: [] };
  }

  // facts about one node, grouped by relation (for tapping a dot)
  async facts(id, per = 6) {
    await this.needAdj([id]);
    const a = this.adj(id), groups = [], ids = [id];
    for (let j = 0; j < a.srel.length && groups.length < 40; j++) {
      const ms = [...a.mem.subarray(a.start[j], a.start[j + 1])].sort((x, y) => this.deg[y] - this.deg[x] || x - y);
      groups.push({ e: a.srel[j], n: ms.length, ms: ms.slice(0, per) });
      ids.push(...ms.slice(0, per));
    }
    await this.needNames(ids);
    groups.sort((x, y) => (x.e & 1) - (y.e & 1) || y.n - x.n);
    return { id, name: this.nm(id), isNew: false, groups: groups.map(g => ({ label: this.edgeLabels[g.e], n: g.n, items: g.ms.map(v => ({ id: v, name: this.nm(v), taught: false })) })) };
  }
}

function matmul(A, B) {
  const C = new Float64Array(DD);
  for (let i = 0; i < D; i++) for (let k = 0; k < D; k++) { const a = A[i * D + k]; if (a === 0) continue; for (let j = 0; j < D; j++) C[i * D + j] += a * B[k * D + j]; }
  return C;
}
function normed(v) { let s = 0; for (const x of v) s += x * x; s = Math.max(Math.sqrt(s), 1e-12); return Float64Array.from(v, x => x / s); }
