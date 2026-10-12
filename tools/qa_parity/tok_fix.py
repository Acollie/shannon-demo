import json, sys
from pathlib import Path
from transformers import AutoTokenizer
from treatygraph import generic
ex = json.loads(Path(sys.argv[1]).read_text()); dd = Path(sys.argv[2])
pars, qs = [], []
for d in ex["docs"]:
    pars += [p for _, p in generic.passages((dd / Path(d["file"]).name).read_text())]
    qs += [q["q"] for q in d["questions"]]
odd = ["Café déjà vu — naïve résumé", "Ⅻ ½ ² ③ x²", "日本語のテキスト and 中文", "tab\there  two   spaces\n\nnew", "emoji 🙂👍 test",
       "U.S.A. $1,000,000.50 (approx.) 3-4 \"quoted\" ‘single’", "İstanbul ΣΟΦΙΑ straße ﬁne", "zero​width﻿bom­soft",
       "ꙮꙮꙮ unknown ꙮ chars", "Mr. O'Neil's co-operation; isn't it?", "    leading and trailing    ", "a" * 120 + " long", "́combining first"]
qa = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
em = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
yn = AutoTokenizer.from_pretrained(str(generic.YESNO))
out = {"reader": [], "embed": [], "yesno": []}
texts = pars + qs + odd
for i, t in enumerate(texts):
    q = qs[i % len(qs)]
    f = qa(q, t, truncation="only_second", max_length=384, return_offsets_mapping=True)
    out["reader"].append({"q": q, "p": t, "ids": f["input_ids"], "type": f["token_type_ids"], "offs": f["offset_mapping"]})
    e = em([t], truncation=True, max_length=256)
    out["embed"].append({"t": t, "ids": e["input_ids"][0]})
    y = yn([t], [q], truncation=True, max_length=320)
    out["yesno"].append({"a": t, "b": q, "ids": y["input_ids"][0]})
# long pair truncation cases for yes/no
long = " ".join(pars[:6])
for q in qs[:5]:
    y = yn([long], [q], truncation=True, max_length=320); out["yesno"].append({"a": long, "b": q, "ids": y["input_ids"][0]})
    y = yn([q], [long], truncation=True, max_length=320); out["yesno"].append({"a": q, "b": long, "ids": y["input_ids"][0]})
    f = qa(q, long, truncation="only_second", max_length=384, return_offsets_mapping=True)
    out["reader"].append({"q": q, "p": long, "ids": f["input_ids"], "type": f["token_type_ids"], "offs": f["offset_mapping"]})
# python-side helpers
out["numbers"] = [[t, generic.numbers_in(t)] for t in texts + ["one hundred and five thousand", "twenty one", "two three", "twenty thirty", "seven hundred and fifty", "one million two hundred thousand", "a hundred", "and five", "1,000 and 5,00"]]
out["words"] = [[t, generic.words(t)] for t in texts]
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False))
yt = json.load(open(generic.YESNO / "tokenizer.json"))
Path(sys.argv[4]).write_text(json.dumps({"unk_id": yt["model"]["unk_id"], "vocab": yt["model"]["vocab"]}, ensure_ascii=False, separators=(",", ":")))
print(len(texts), "texts")
