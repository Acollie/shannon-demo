// Model inference for Shannon QA. `ort` is onnxruntime-web (browser) or onnxruntime-node (parity tests);
// `fetchBytes(url, onBytes)` returns the file as a Uint8Array (the browser version caches and reports progress).

import { WordPiece, Unigram } from "./tokenizers.js";

export class Engine {
  constructor(ort, { embed, reader, yesno, vocab, unigram }) {
    this.ort = ort;
    this.files = { embed, reader, yesno };
    this.wp = new WordPiece(vocab);
    this.ug = new Unigram(unigram.vocab, unigram.unk_id);
  }

  async init(fetchBytes, opts = {}) {
    this.s = {};
    for (const k of ["embed", "reader", "yesno"]) {
      const bytes = await fetchBytes(this.files[k], k);
      this.s[k] = await this.ort.InferenceSession.create(bytes, opts);
    }
    return this;
  }

  feeds(sess, rows, types) {
    const B = rows.length, L = Math.max(...rows.map(r => r.length));
    const ids = new BigInt64Array(B * L), mask = new BigInt64Array(B * L), tt = new BigInt64Array(B * L);
    rows.forEach((r, b) => r.forEach((v, i) => {
      ids[b * L + i] = BigInt(v); mask[b * L + i] = 1n; tt[b * L + i] = BigInt(types ? types[b][i] : 0);
    }));
    const all = {
      input_ids: new this.ort.Tensor("int64", ids, [B, L]),
      attention_mask: new this.ort.Tensor("int64", mask, [B, L]),
      token_type_ids: new this.ort.Tensor("int64", tt, [B, L]),
    };
    const f = {};
    for (const n of sess.inputNames) f[n] = all[n];
    return [f, B, L];
  }

  // MiniLM sentence vectors (mean-pooled, unit length; pooling is inside the ONNX graph)
  async embed(texts, onProgress, batch = 8) {
    const out = [];
    const sess = this.s.embed;
    for (let i = 0; i < texts.length; i += batch) {
      const rows = texts.slice(i, i + batch).map(t => this.wp.single(t, 256));
      const [f, B] = this.feeds(sess, rows.map(r => r.ids), rows.map(r => r.type));
      const r = await sess.run(f);
      const d = r[sess.outputNames[0]].data;
      const D = d.length / B;
      for (let b = 0; b < B; b++) out.push(d.slice(b * D, (b + 1) * D));
      if (onProgress) onProgress(Math.min(i + batch, texts.length), texts.length);
    }
    return out;
  }

  async reader(ids, type) {
    const sess = this.s.reader;
    const [f] = this.feeds(sess, [ids], [type]);
    const r = await sess.run(f);
    return { st: r[sess.outputNames[0]].data, en: r[sess.outputNames[1]].data };
  }

  // YesNo.p_yes: (passage, question) pairs -> entailment / (entailment + contradiction)
  async pYes(passages, q) {
    const sess = this.s.yesno;
    const rows = passages.map(p => this.ug.pair(p, q, 320));
    const [f, B] = this.feeds(sess, rows, null);
    const r = await sess.run(f);
    const lg = r[sess.outputNames[0]].data;
    const C = lg.length / B;
    const out = [];
    for (let b = 0; b < B; b++) {
      const z = Array.from(lg.slice(b * C, (b + 1) * C));
      const mx = Math.max(...z);
      const e = z.map(v => Math.exp(v - mx));
      const s = e.reduce((a, v) => a + v, 0);
      const p = e.map(v => Math.fround(v / s));
      out.push(Math.fround(p[1] / Math.fround(p[1] + p[0] + 1e-9)));
    }
    return out;
  }
}
