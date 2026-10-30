// BERT-uncased WordPiece tokenizer (the MiniLM vocab Shannon uses for both encoder and decoder), plus the
// detokeniser used by tools/build_data.py, so text on the site reads the same as in the data files.
export class WordPiece {
  constructor(vocabText) {
    this.toks = vocabText.split("\n");
    if (this.toks[this.toks.length - 1] === "") this.toks.pop();
    this.id = new Map(this.toks.map((t, i) => [t, i]));
    this.unk = this.id.get("[UNK]");
  }

  static isPunct(ch) {
    const cp = ch.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(ch);
  }

  static isCJK(cp) {
    return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) ||
      (cp >= 0x2a700 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) || (cp >= 0xf900 && cp <= 0xfaff) ||
      (cp >= 0x2f800 && cp <= 0x2fa1f);
  }

  basic(text) {
    text = text.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
    const words = [];
    let cur = "";
    const flush = () => { if (cur) { words.push(cur); cur = ""; } };
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (cp === 0 || cp === 0xfffd || (/\p{Cc}|\p{Cf}/u.test(ch) && !/\s/.test(ch))) continue;
      if (/\s/.test(ch)) { flush(); continue; }
      if (WordPiece.isPunct(ch) || WordPiece.isCJK(cp)) { flush(); words.push(ch); continue; }
      cur += ch;
    }
    flush();
    return words;
  }

  encode(text) {
    const ids = [];
    for (const w of this.basic(text)) {
      const chars = Array.from(w);
      if (chars.length > 100) { ids.push(this.unk); continue; }
      const sub = [];
      let start = 0, bad = false;
      while (start < chars.length) {
        let end = chars.length, found = -1;
        while (start < end) {
          let s = chars.slice(start, end).join("");
          if (start > 0) s = "##" + s;
          const i = this.id.get(s);
          if (i !== undefined) { found = i; break; }
          end--;
        }
        if (found < 0) { bad = true; break; }
        sub.push(found);
        start = end;
      }
      if (bad) ids.push(this.unk); else ids.push(...sub);
    }
    return ids;
  }

  decode(ids) {
    const out = [];
    for (const i of ids) {
      const p = this.toks[i] ?? "[UNK]";
      if (p.startsWith("##") && out.length) out[out.length - 1] += p.slice(2);
      else out.push(p);
    }
    let s = out.join(" ");
    s = s.replace(/ ([.,;:!?%)\]}])/g, "$1");
    s = s.replace(/([(\[{$]) /g, "$1");
    s = s.replace(/ ' (s|t|re|ve|ll|d|m)\b/g, "'$1");
    s = s.replace(/ - /g, "-");
    return s;
  }
}
