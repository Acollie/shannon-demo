"""GLM Pocket: choose the graph-confidence rule offline from Mintaka DEV (never sealed data).
Input: tools/glm_parity/pocket_scores_dev.json (tools/glm_parity/pocket_scores.mjs: the browser engine's top path score,
answer type and correctness for each of the 2,000 dev questions). Output: assets/pocket/calibration.json.
Rule: keep the graph's answer when its answer type is in TYPES and its path score >= CUT. Target was >= ~70% precision;
no cut reaches that (see the curve in the output), so the cut with the best precision at >= 30 kept answers is used."""
import json, pathlib
SITE = pathlib.Path(__file__).resolve().parents[2]
rows = json.loads((SITE / "tools/glm_parity/pocket_scores_dev.json").read_text())["rows"]
OP2TYPE = {"set": "pick", "count": "count", "yesno": "yes / no", "argmax": "largest", "argmin": "smallest", "compare": "compare"}
def stat(rs, cut, ops):
    s = [r for r in rs if r["ok"] and r["op"] in ops and r["score"] >= cut]
    return len(s), sum(r["hit"] for r in s)
curve_all = []
for cut in [40, 45, 48, 50, 51, 52, 53, 54, 55]:
    n, h = stat(rows, cut, set(OP2TYPE))
    curve_all.append({"cut": cut, "kept": n, "right": h, "precision": round(h / max(n, 1), 3)})
curve_pick = []
for cut in [45, 48, 50, 51, 52, 53, 54, 55]:
    n, h = stat(rows, cut, {"set"})
    curve_pick.append({"cut": cut, "kept": n, "right": h, "precision": round(h / max(n, 1), 3)})
best = max((c for c in curve_pick if c["kept"] >= 30), key=lambda c: (c["precision"], c["kept"]))
CUT, OPS = best["cut"], ["set"]
A = [r for i, r in enumerate(rows) if i % 2 == 0]; B = [r for i, r in enumerate(rows) if i % 2]
(na, ha), (nb, hb) = stat(A, CUT, set(OPS)), stat(B, CUT, set(OPS))
out = {
    "pack": "general", "threshold": CUT, "types": [OP2TYPE[o] for o in OPS],
    "precision": best["precision"], "precision_pct": round(100 * best["precision"]), "kept": best["kept"], "right": best["right"],
    "coverage_pct": round(100 * best["kept"] / len(rows), 1), "n_dev": len(rows), "dev_hits_all": sum(r["hit"] for r in rows),
    "split_halves": {"even_ids": [na, ha], "odd_ids": [nb, hb]},
    "curve_all_types": curve_all, "curve_pick_only": curve_pick,
    "note": ("Graph answers are kept only for 'pick' answers with path score >= %s on the general pack. Calibrated offline on Mintaka DEV "
             "(2,000 questions; tools/glm_parity/pocket_scores_dev.json from tools/glm_parity/pocket_scores.mjs, the page's own engine). "
             "Precision of kept answers: %d/%d = %.1f%% (in-sample on dev; split halves %d/%d and %d/%d, so it is noisy). The ~70%% target is "
             "NOT reached by any score cut: the best all-types cut is about 50-56%%, and count / largest / yes-no answers stay near their base rate "
             "at any score. Everything below the cut goes to the on-device LM when it is on, or is shown with a low-confidence note. "
             "The movies pack uses a different scorer scale, so there any committed answer is kept. No sealed data was used.")
            % (CUT, best["right"], best["kept"], 100 * best["precision"], ha, na, hb, nb),
}
(SITE / "assets/pocket/calibration.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ["threshold", "types", "precision", "kept", "coverage_pct", "split_halves"]}))
