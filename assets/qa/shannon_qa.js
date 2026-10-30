// Shannon QA: a line-by-line port of treaty-graph's SystemX (src/treatygraph/generic.py) to the browser.
//   passages -> IDF overlap + MiniLM retrieval (top 3 each) -> minilm-squad2 spans in generic forms
//   (span, list, number, yes/no, none) -> DeBERTa yes/no -> question-form model + log-linear gate.
// Model inference is injected (see engine.js), so this file is pure logic and runs in Node for parity tests.

import { argsort } from "./npsort.js";

export const FORMS = ["list", "span", "number", "bool", "null"];

// Python str.isspace() / str.split() whitespace
const PYWS = "[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const RE_WS = new RegExp(PYWS + "+", "u");
const RE_BLOCK = new RegExp(`([^]+?)(?:\\n${PYWS}*\\n|$)`, "gu");
const RE_LSTRIP = new RegExp("^" + PYWS + "*", "u");
const RE_SENT = /(?<=[.!?;:]) +/u;

export const pysplit = s => s.split(RE_WS).filter(Boolean);

// -- hybrid.py tables ------------------------------------------------------------------------------
const NUMWORDS = {};
"zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
  .split(" ").forEach((w, i) => { NUMWORDS[w] = i; });
Object.assign(NUMWORDS, { thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 });
const SCALES = { hundred: 100, thousand: 1000, million: 1000000 };
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

// generic.numbers_in: (value, start, end) of numbers in digits or English words
export function numbersIn(text) {
  const out = [];
  const toks = [...text.matchAll(/[0-9]+(?:,[0-9]{3})*|[A-Za-z]+/g)].map(m => ({ g: m[0], s: m.index, e: m.index + m[0].length }));
  let i = 0;
  while (i < toks.length) {
    const w = toks[i].g.toLowerCase();
    if (/[0-9]/.test(w[0])) { out.push([Number(w.replace(/,/g, "")), toks[i].s, toks[i].e]); i++; continue; }
    if (!has(NUMWORDS, w)) { i++; continue; }
    let total = 0, cur = 0, j = i, end = toks[i].e, last = null;
    while (j < toks.length) {
      const x = toks[j].g.toLowerCase();
      if (has(NUMWORDS, x)) {
        const v = NUMWORDS[x];
        if (last === "unit" || (last === "tens" && v >= 10)) break;
        cur += v;
        last = v >= 20 ? "tens" : "unit";
      } else if (has(SCALES, x) && last !== null) {
        cur = Math.max(cur, 1) * SCALES[x];
        if (SCALES[x] >= 1000) { total += cur; cur = 0; }
        last = "scale";
      } else if (x === "and" && last === "scale" && j + 1 < toks.length && has(NUMWORDS, toks[j + 1].g.toLowerCase())) {
        // pass
      } else break;
      end = toks[j].e;
      j++;
    }
    out.push([total + cur, toks[i].s, end]);
    i = j;
  }
  return out;
}

// generic.words: \w+ on the lower-cased text
export const words = s => s.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];

// generic.passages: (offset, passage) from the document's own blocks; lengths only
export function passages(text, maxWords = 120, minWords = 12) {
  const out = [];
  let carry = "";
  for (const m of text.matchAll(RE_BLOCK)) {
    const g = m[1];
    const block = pysplit(g).join(" ");
    if (!block) continue;
    if (block.split(" ").length < minWords) { carry = (carry + " " + block).trim(); continue; }
    const sents = block.split(RE_SENT);
    let cur = carry;
    const start = m.index + g.match(RE_LSTRIP)[0].length;
    carry = "";
    for (const s of sents) {
      if (cur && (cur + " " + s).split(" ").filter(Boolean).length > maxWords) { out.push([start, cur]); cur = ""; }
      cur = (cur + " " + s).trim();
    }
    if (cur) out.push([start, cur]);
  }
  if (carry) out.push([text.length, carry]);
  return out;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const f32 = Math.fround;

// -- the learned parts ---------------------------------------------------------------------------------
export class Gate {
  constructor(d) {          // gate_x.json
    this.fm = { mu: d.form.mu, sd: d.form.sd, W: d.form.W };
    this.ix = new Map(d.gate.names.map((k, i) => [k, i]));
    this.w = d.gate.w; this.mu = d.gate.mu; this.sd = d.gate.sd;
  }

  // XForm.logp: log P(form | question embedding)
  logp(qv) {
    const { mu, sd, W } = this.fm;
    const z = FORMS.map((_, c) => {
      let s = W[mu.length][c];
      for (let i = 0; i < mu.length; i++) s += ((qv[i] - mu[i]) / sd[i]) * W[i][c];
      return s;
    });
    const mx = Math.max(...z);
    const zz = z.map(v => v - mx);
    const lse = Math.log(zz.reduce((a, v) => a + Math.exp(v), 0));
    return zz.map(v => v - lse);
  }

  // generic.featurize + Gate.pick, returning every candidate's score and its feature contributions
  score(cands, lpf) {
    return cands.map(c => {
      const f = { [`${c.form}:bias`]: 1.0, lpf: lpf[FORMS.indexOf(c.form)] };
      for (const [k, v] of Object.entries(c.feats)) f[`${c.form}:${k}`] = Number(v);
      let s = 0;
      const contrib = [];
      for (const [k, v] of Object.entries(f)) {
        const i = this.ix.get(k);
        if (i === undefined || v === 0) continue;
        const x = (v - this.mu[i]) / this.sd[i];
        s += x * this.w[i];
        contrib.push([k, x * this.w[i]]);
      }
      return { s, contrib };
    });
  }
}

// -- evidence over one document ---------------------------------------------------------------------
export class Evidence {
  constructor(engine, { k = 3, spans = 6 } = {}) {
    this.e = engine; this.k = k; this.nspans = spans;
  }

  async read(text, onProgress) {
    this.text = text;
    this.pars = passages(text);
    const df = new Map();
    for (const [, p] of this.pars) for (const w of new Set(words(p))) df.set(w, (df.get(w) || 0) + 1);
    const n = this.pars.length;
    this.idf = new Map([...df].map(([w, c]) => [w, Math.log((n + 1) / (c + 0.5))]));
    this.pw = this.pars.map(([, p]) => new Set(words(p)));
    this.pv = await this.e.embed(this.pars.map(([, p]) => p), onProgress);
  }

  async retrieve(q) {
    const [qv] = await this.e.embed([q]);
    const qw = [...new Set(words(q))];
    const cos = this.pv.map(v => f32(dot(v, qv)));
    const ovl = this.pw.map(pw => qw.reduce((s, w) => (pw.has(w) ? s + (this.idf.get(w) || 0) : s), 0));
    const byE = argsort(cos.map(x => -x)).slice(0, this.k);
    const byO = argsort(ovl.map(x => -x)).slice(0, this.k);
    const picked = [...new Set([...byO, ...byE])];
    return { picked, cos, ovl, topE: new Set(byE), topO: new Set(byO), byE, byO, qv, qw };
  }

  async qaSpans(q, par) {
    const x = this.e.wp.pairOnlySecond(q, par, 384);
    const { st, en } = await this.e.reader(x.ids, x.type);
    const nul = f32(st[0] + en[0]);
    const ctx = [];
    x.type.forEach((t, i) => { if (t === 1 && x.offs[i] && x.offs[i][1] > 0) ctx.push(i); });
    const topS = [...ctx].sort((a, b) => st[b] - st[a]).slice(0, 12);
    const topE = [...ctx].sort((a, b) => en[b] - en[a]).slice(0, 12);
    const cands = [];
    for (const i of topS) for (const j of topE) if (i <= j && j < i + 30) cands.push([f32(st[i] + en[j]), x.offs[i][0], x.offs[j][1]]);
    cands.sort((u, v) => v[0] - u[0] || v[1] - u[1] || v[2] - u[2]);
    const out = [], seen = new Set();
    for (const [sc, a, b] of cands) {
      const s = par.slice(a, b);
      if (seen.has(s.toLowerCase())) continue;
      seen.add(s.toLowerCase());
      out.push([sc - nul, s, a, b]);
      if (out.length >= this.nspans) break;
    }
    return { nul, spans: out };
  }

  async candidates(q) {
    const R = await this.retrieve(q);
    const { picked, cos, ovl, topE, topO, qw } = R;
    const unknown = qw.filter(w => !this.idf.has(w));
    const omax = Math.max(Math.max(...ovl), 1e-6);
    const cands = [];
    let bestMargin = -50.0, bestPar = picked[0];
    const read = [];
    for (const pi of picked) {
      const [pos, par] = this.pars[pi];
      const { nul, spans } = await this.qaSpans(q, par);
      read.push({ pi, nul, spans });
      const pf = { cos: cos[pi], ovl: ovl[pi] / omax, emb_top: +(pi === picked[picked.length - 1] && topE.has(pi)),
        ovl_top: +topO.has(pi), in_both: +(topE.has(pi) && topO.has(pi)) };
      spans.forEach(([margin, s, a, b], r) => {
        if (margin > bestMargin) { bestMargin = margin; bestPar = pi; }
        const base = { ...pf, qa: margin, qa_pos: +(margin > 0), qa_r0: +(r === 0), len: Math.min(pysplit(s).length, 30) / 10 };
        const src = { pi, a, b };
        cands.push({ form: "span", answer: s, pos, feats: base, src });
        cands.push({ form: "list", answer: [s], pos, feats: { ...base, nb: 1 }, src });
        for (const [n, na, nb] of numbersIn(s).slice(0, 1))
          cands.push({ form: "number", answer: n, pos, feats: { ...base, digit: +/[0-9]/.test(s) }, src: { pi, a: a + na, b: a + nb, span: [a, b] } });
      });
      const m = Math.min(spans.length, 4);
      for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
        const [m1, s1, a1, b1] = spans[i], [m2, s2, a2, b2] = spans[j];
        if (b1 <= a2 || b2 <= a1) {
          const [first, second] = a1 < a2 ? [s1, s2] : [s2, s1];
          cands.push({ form: "list", answer: [first, second], pos,
            feats: { ...pf, qa: (m1 + m2) / 2, qa_pos: +(Math.min(m1, m2) > 0), nb: 2,
              len: Math.min(pysplit(s1).length + pysplit(s2).length, 30) / 10 },
            src: { pi, parts: a1 < a2 ? [[a1, b1], [a2, b2]] : [[a2, b2], [a1, b1]] } });
        }
      }
    }
    const topIdx = [...picked.slice(0, 2), bestPar];
    const py = await this.e.pYes(topIdx.map(pi => this.pars[pi][1]), q);
    const cosMax = Math.max(...cos);
    const yfe = { cos_max: cosMax, qa_best: bestMargin };
    const mean = py.reduce((a, b) => a + b, 0) / py.length;
    const p0 = this.pars[picked[0]][0];
    cands.push({ form: "bool", answer: true, pos: p0, src: { pi: picked[0] },
      feats: { ...yfe, p_yes0: py[0], p_yes_max: Math.max(...py), p_yes_mean: mean, yes_bias: 1.0 } });
    cands.push({ form: "bool", answer: false, pos: p0, src: { pi: picked[0] },
      feats: { ...yfe, p_no0: 1 - py[0], p_no_max: 1 - Math.min(...py), p_no_mean: 1 - mean, no_bias: 1.0 } });
    cands.push({ form: "null", answer: null, pos: null, src: null,
      feats: { qa_best: bestMargin, qa_best_pos: +(bestMargin > 0), cos_max: cosMax,
        unknown: unknown.length / Math.max(qw.length, 1), unknown_any: +(unknown.length > 0) } });
    return { cands, R, read, py, topIdx };
  }
}

// -- the system -----------------------------------------------------------------------------------------
export class ShannonQA {
  constructor(engine, gateJson) {
    this.e = engine;
    this.gate = new Gate(gateJson);
    this.ev = new Evidence(engine);
  }

  async read(text, onProgress) { await this.ev.read(text, onProgress); }

  async ask(question) {
    const t0 = performance.now();
    const { cands, R, read, py, topIdx } = await this.ev.candidates(question);
    const lpf = this.gate.logp(R.qv);
    const sc = this.gate.score(cands, lpf);
    let best = 0;
    sc.forEach((x, i) => { if (x.s > sc[best].s) best = i; });
    const sorted = sc.map(x => x.s).sort((a, b) => b - a);
    const margin = sorted.length > 1 ? sorted[0] - sorted[1] : Infinity;
    const c = cands[best];
    return {
      question, answer: c.answer, form: c.form, chosen: best, margin,
      cite: c.pos !== null && c.form !== "null" ? [c.pos] : [],
      cands: cands.map((x, i) => ({ ...x, score: sc[i].s, contrib: sc[i].contrib })),
      retrieval: { picked: R.picked, byE: R.byE, byO: R.byO, cos: R.cos, ovl: R.ovl, qw: R.qw },
      read, py, topIdx, lpf, ms: performance.now() - t0,
    };
  }
}
