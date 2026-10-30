import fs from "fs";
const SITE = "/Users/alexcollie/Documents/code/github/shannon-demo";
const { WordPiece, Unigram } = await import(SITE + "/assets/qa/tokenizers.js");
const { numbersIn, words, passages } = await import(SITE + "/assets/qa/shannon_qa.js");
const fx = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const wp = new WordPiece(fs.readFileSync(SITE + "/models/20m-fineweb/vocab.txt", "utf8"));
const uv = JSON.parse(fs.readFileSync(SITE + "/models/qa/yesno_vocab.json", "utf8"));
const ug = new Unigram(uv.vocab, uv.unk_id);
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const cp2u = (s, i) => Array.from(s).slice(0, i).join("").length;   // python code-point offset -> utf16
let bad = { reader: 0, offs: 0, embed: 0, yesno: 0, numbers: 0, words: 0 };
for (const r of fx.reader) {
  const x = wp.pairOnlySecond(r.q, r.p, 384);
  if (!eq(x.ids, r.ids) || !eq(x.type, r.type)) { if (bad.reader++ < 3) console.log("READER", r.p.slice(0, 60), "\n", x.ids.slice(0, 40).join(","), "\n", r.ids.slice(0, 40).join(",")); continue; }
  const po = r.offs.map((o, i) => (r.type[i] === 1 && o[1] > 0 ? [cp2u(r.p, o[0]), cp2u(r.p, o[1])] : null));
  const jo = x.offs.map((o, i) => (x.type[i] === 1 && o && o[1] > 0 ? o : null));
  if (!eq(po, jo)) { if (bad.offs++ < 3) { const k = po.findIndex((o, i) => !eq(o, jo[i])); console.log("OFFS", r.p.slice(0, 50), k, po[k], jo[k]); } }
}
for (const r of fx.embed) { const x = wp.single(r.t, 256); if (!eq(x.ids, r.ids) && bad.embed++ < 3) console.log("EMBED", r.t.slice(0, 50)); }
for (const r of fx.yesno) { const x = ug.pair(r.a, r.b, 320); if (!eq(x, r.ids)) { if (bad.yesno++ < 5) console.log("YESNO", JSON.stringify(r.a.slice(0, 50)), "\n", x.slice(0, 30).join(","), "\n", r.ids.slice(0, 30).join(","), x.length, r.ids.length); } }
for (const [t, n] of fx.numbers) { const got = numbersIn(t).map(([v, a, b]) => [v, Array.from(t.slice(0, a)).length, Array.from(t.slice(0, b)).length]); if (!eq(got, n) && bad.numbers++ < 3) console.log("NUM", t.slice(0, 60), JSON.stringify(got), JSON.stringify(n)); }
for (const [t, w] of fx.words) if (!eq(words(t), w) && bad.words++ < 3) console.log("WORDS", t.slice(0, 60), words(t).slice(0, 10), w.slice(0, 10));
console.log({ n_reader: fx.reader.length, n_embed: fx.embed.length, n_yesno: fx.yesno.length, n_num: fx.numbers.length, bad });
