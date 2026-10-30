"""Re-label the k-means regions of map.json from dictionary.json (no model needed).

Each region's label = its 3 most frequent distinctive words ("w" in dictionary.json), weighted by code usage,
drops stop-words (also from each code's "w" list), then penalised by how many regions also use the word, so labels differ between regions.

  python3 tools/relabel_regions.py models/<model-id>
"""
import json
import math
import sys
from collections import Counter

import os
import re

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_data.py")).read()
STOP = set(re.search(r'STOP = set\("""(.*?)"""', _src, re.S).group(1).split())  # same list as build_data.py (no torch import)

STOP = STOP | {"the", "and", "for", "are", "was", "with", "were", "that", "this", "has", "had", "not", "but", "its", "his", "her", "they", "you", "can", "who", "one", "all", "our", "out", "how", "why", "use"}

d = sys.argv[1]
mp = json.load(open(f"{d}/map.json"))
dic = json.load(open(f"{d}/dictionary.json"))
R = len(mp["regions"])
per = [Counter() for _ in range(R)]
for c, r in enumerate(mp["region"]):
    e = dic[c]
    for k, w in enumerate(e.get("w", [])):
        if w.isalpha() and len(w) > 2 and w not in STOP:
            per[r][w] += math.log1p(e["n"]) / (1 + k)
df = Counter(w for cnt in per for w in cnt)
for r, cnt in enumerate(per):
    sc = sorted(((v * math.log(1 + R / df[w]), w) for w, v in cnt.items()), reverse=True)
    mp["regions"][r]["label"] = " · ".join(w for _, w in sc[:3])
for e in dic:
    if "w" in e:
        e["w"] = [w for w in e["w"] if w not in STOP]
json.dump(dic, open(f"{d}/dictionary.json", "w"), separators=(",", ":"), ensure_ascii=False)
json.dump(mp, open(f"{d}/map.json", "w"), separators=(",", ":"))
print([r["label"] for r in mp["regions"]])
