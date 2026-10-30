// Tokenisers for Shannon QA, written to reproduce the Hugging Face "tokenizers" output the Python system uses.
//   WordPiece: BERT-uncased (all-MiniLM-L6-v2 and minilm-uncased-squad2 share this vocab), with character offsets.
//   Unigram:   DeBERTa-v3 (the yes/no cross-encoder), from the vocab and scores in its tokenizer.json.
// Offsets are UTF-16 indices into the original string, so text.slice(a, b) gives the same span Python's par[a:b] does.

const PUNCT = /\p{P}/u;
const WS = /[\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u;   // Rust char::is_whitespace
const CTRL = /[\p{Cc}\p{Cf}]/u;
const MARK = /\p{Mn}/gu;

function isCJK(cp) {
  return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) || (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f);
}

function isPunct(ch) {
  const cp = ch.codePointAt(0);
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return PUNCT.test(ch);
}

export class WordPiece {
  constructor(vocabText) {
    const toks = vocabText.split("\n");
    if (toks[toks.length - 1] === "") toks.pop();
    this.id = new Map(toks.map((t, i) => [t, i]));
    this.unk = this.id.get("[UNK]");
    this.cls = this.id.get("[CLS]");
    this.sep = this.id.get("[SEP]");
  }

  // words as arrays of [normalised char, origStart, origEnd]
  words(text) {
    const words = [];
    let cur = [];
    const flush = () => { if (cur.length) { words.push(cur); cur = []; } };
    let i = 0;
    for (const ch of text) {
      const a = i, b = i + ch.length;
      i = b;
      const cp = ch.codePointAt(0);
      if (WS.test(ch)) { flush(); continue; }
      if (cp === 0 || cp === 0xfffd || CTRL.test(ch)) continue;
      if (isCJK(cp)) { flush(); words.push([[ch, a, b]]); continue; }
      const norm = ch.normalize("NFD").replace(MARK, "").toLowerCase();
      for (const c of norm) {
        if (WS.test(c)) { flush(); continue; }
        if (isPunct(c)) { flush(); words.push([[c, a, b]]); continue; }
        cur.push([c, a, b]);
      }
    }
    flush();
    return words;
  }

  // -> {ids, offs}  (no special tokens)
  encode(text) {
    const ids = [], offs = [];
    for (const w of this.words(text)) {
      const a0 = w[0][1], b0 = w[w.length - 1][2];
      if (w.length > 100) { ids.push(this.unk); offs.push([a0, b0]); continue; }
      const sub = [];
      let start = 0, bad = false;
      while (start < w.length) {
        let end = w.length, found = -1;
        while (start < end) {
          let s = "";
          for (let k = start; k < end; k++) s += w[k][0];
          if (start > 0) s = "##" + s;
          const id = this.id.get(s);
          if (id !== undefined) { found = id; break; }
          end--;
        }
        if (found < 0) { bad = true; break; }
        sub.push([found, w[start][1], w[end - 1][2]]);
        start = end;
      }
      if (bad) { ids.push(this.unk); offs.push([a0, b0]); }
      else for (const [id, a, b] of sub) { ids.push(id); offs.push([a, b]); }
    }
    return { ids, offs };
  }

  // [CLS] text [SEP], truncated to maxLen
  single(text, maxLen) {
    const e = this.encode(text);
    const ids = [this.cls, ...e.ids.slice(0, maxLen - 2), this.sep];
    return { ids, type: ids.map(() => 0) };
  }

  // [CLS] q [SEP] p [SEP] with truncation="only_second"; offs are into p (null for specials / question)
  pairOnlySecond(q, p, maxLen) {
    const eq = this.encode(q), ep = this.encode(p);
    const room = Math.max(0, maxLen - 3 - eq.ids.length);
    const pid = ep.ids.slice(0, room), poff = ep.offs.slice(0, room);
    const ids = [this.cls, ...eq.ids, this.sep, ...pid, this.sep];
    const type = ids.map((_, i) => (i < eq.ids.length + 2 ? 0 : 1));
    const offs = [null, ...eq.ids.map(() => null), null, ...poff, null];
    return { ids, type, offs };
  }
}

// DeBERTa-v3 Unigram: Replace(\s{2,}|[\n\r\t] -> " "), NFC, strip right; Metaspace(prepend always, split); Viterbi.
export class Unigram {
  constructor(vocab, unkId) {           // vocab: [[piece, score], ...]
    this.piece = new Map();
    this.maxLen = 1;
    let minScore = Infinity;
    vocab.forEach(([p, s], i) => {
      this.piece.set(p, [i, s]);
      const n = Array.from(p).length;
      if (n > this.maxLen) this.maxLen = n;
      if (s < minScore) minScore = s;
    });
    this.unkId = unkId;
    this.unkScore = minScore - 10.0;
    this.cls = this.piece.get("[CLS]")[0];
    this.sep = this.piece.get("[SEP]")[0];
    this.pad = this.piece.get("[PAD]")[0];
  }

  normalize(text) {
    let t = text.replace(/\s{2,}|[\n\r\t]/gu, " ").normalize("NFC");
    return t.replace(/\s+$/u, "");
  }

  pretokenize(t) {
    t = t.replace(/ /g, "▁");
    if (!t.startsWith("▁")) t = "▁" + t;
    const out = [];
    let cur = "";
    for (const ch of t) {
      if (ch === "▁" && cur) { out.push(cur); cur = ""; }
      cur += ch;
    }
    if (cur) out.push(cur);
    return out;
  }

  viterbi(word) {
    const ch = Array.from(word);
    const n = ch.length;
    const best = new Array(n + 1).fill(-Infinity), back = new Array(n + 1).fill(null);
    best[0] = 0;
    // nodes ending at e, in the order the Rust lattice inserts them (begin ascending)
    const ends = Array.from({ length: n + 1 }, () => []);
    for (let b = 0; b < n; b++) {
      let s = "", single = false;
      for (let l = 1; l <= this.maxLen && b + l <= n; l++) {
        s += ch[b + l - 1];
        const hit = this.piece.get(s);
        if (hit) { ends[b + l].push([b, hit[0], hit[1]]); if (l === 1) single = true; }
      }
      if (!single) ends[b + 1].push([b, this.unkId, this.unkScore]);
    }
    for (let e = 1; e <= n; e++) {
      for (const [b, id, sc] of ends[e]) {
        if (best[b] === -Infinity) continue;
        const v = best[b] + sc;
        if (back[e] === null || v > best[e]) { best[e] = v; back[e] = [b, id]; }
      }
    }
    const ids = [];
    let e = n;
    while (e > 0) { const [b, id] = back[e]; ids.push(id); e = b; }
    ids.reverse();
    const fused = [];
    for (const id of ids) if (!(id === this.unkId && fused[fused.length - 1] === this.unkId)) fused.push(id);
    return fused;
  }

  encode(text) {
    const t = this.normalize(text);
    if (!t) return [];
    return this.pretokenize(t).flatMap(w => this.viterbi(w));
  }

  // [CLS] a [SEP] b [SEP] with LongestFirst truncation to maxLen
  pair(a, b, maxLen) {
    let ea = this.encode(a), eb = this.encode(b);
    const m = maxLen - 3;
    if (ea.length + eb.length > m) {
      let n1 = ea.length, n2 = eb.length;
      const swap = n1 > n2;
      if (swap) [n1, n2] = [n2, n1];
      if (n1 > Math.floor(m / 2)) { n1 = Math.floor(m / 2); n2 = m - n1; }
      else n2 = m - n1;
      if (swap) [n1, n2] = [n2, n1];
      ea = ea.slice(0, n1); eb = eb.slice(0, n2);
    }
    return [this.cls, ...ea, this.sep, ...eb, this.sep];
  }
}
