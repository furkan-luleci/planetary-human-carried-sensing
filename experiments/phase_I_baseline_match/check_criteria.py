"""Evaluate the criteria declared in docs/phase_reports/PHASE_I_baseline_match.md
(2026-11-20, before the matched trials were analysed) for using the the earlier study
index thresholds with this body model:

1. On the Earth baseline check (standing at mid-span; walking at 1.5 m/s), B1
   is classified observable for both devices in both activities.
2. No contradiction with the threshold-free measures (same records, both
   devices, all situations and planets, modes B1, B2, T1, B3, L1, T2): no mode
   that is recovered blind and stands >= 10 dB above its matched background is
   classified unobservable, and no mode that is not recovered blind and stands
   < 3 dB above its background is classified observable.

Also descriptive: the cross-table of index class against visibility and blind
recovery, and the rank correlation between index and visibility.

Reads outputs/tables/outcomes.json and outputs/tables/legacy_oi_baseline.json;
writes outputs/tables/threshold_check.json.

    python experiments/phase_I_baseline_match/check_criteria.py
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMMON = ["B1", "B2", "T1", "B3", "L1", "T2"]
DEV = {"wrist": "watch", "thigh": "phone"}
SHORT = {"observable": "O", "marginally observable": "M", "unobservable": "U"}


def main():
    O = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json")))
    L = json.load(open(os.path.join(ROOT, "outputs", "tables", "legacy_oi_baseline.json")))
    labels = [m["label"] for m in O["modes"]]
    res = dict(criteria_declared="docs/phase_reports/PHASE_I_baseline_match.md (2026-11-20 00:30 UTC, commit 14fc883)")
    # criterion 1
    c1 = []
    for sid in ("baseline_stand", "baseline_walk"):
        r = L["data"].get(sid, {}).get("Earth")
        for dev in ("watch", "phone"):
            cl = r["table3"][dev]["pmode"]["B1"]["band"] if r else None
            c1.append(dict(case=sid, device=dev, B1_class=cl, B1_oi=(r["table3"][dev]["pmode"]["B1"]["oi"] if r else None),
                           ok=cl == "observable"))
    res["criterion1"] = dict(cells=c1, holds=bool(c1) and all(c["ok"] for c in c1) and len(c1) == 4)
    # criterion 2 and the cross-table
    cells, bad = [], []
    for sid, byp in O["data"].items():
        for p, d in byp.items():
            lr = L["data"].get(sid, {}).get(p)
            if not lr:
                continue
            for sk, dev in DEV.items():
                rec = d[sk]["rec"]
                for m in COMMON:
                    i = labels.index(m)
                    f_m = O["modes"][i]["f"]
                    found = any(abs(q[0] - f_m) < 1e-6 for q in rec["pairs"])
                    vis = d[sk]["vis_db"][i]
                    pm = lr["table3"][dev]["pmode"][m]
                    c = dict(case=sid, planet=p, device=dev, mode=m, found=bool(found), vis_db=float(vis),
                             oi=float(pm["oi"]), cls=SHORT[pm["band"]])
                    cells.append(c)
                    if (found and vis >= 10.0 and c["cls"] == "U") or (not found and vis < 3.0 and c["cls"] == "O"):
                        bad.append(c)
    act = lambda c: "walking" if c["case"].startswith(("walk", "baseline_walk")) else "standing"
    res["criterion2"] = dict(cells=len(cells), contradictions=bad, holds=len(bad) == 0,
                             cells_by_activity={a: sum(act(c) == a for c in cells) for a in ("standing", "walking")},
                             contradictions_by_activity={a: sum(act(c) == a for c in bad) for a in ("standing", "walking")})
    # descriptive
    def vbin(v):
        return "< 3 dB" if v < 3 else ("3-10 dB" if v < 10 else ">= 10 dB")
    tab = {}
    for c in cells:
        k = f"{vbin(c['vis_db'])}, {'found' if c['found'] else 'not found'}"
        tab.setdefault(k, {"U": 0, "M": 0, "O": 0})[c["cls"]] += 1
    res["cross_table"] = tab
    if len(cells) > 2:
        rho, pv = spearmanr([c["oi"] for c in cells], [c["vis_db"] for c in cells])
        res["spearman_oi_vs_visibility"] = dict(rho=float(rho), p=float(pv), n=len(cells))
    walk = [c for c in cells if c["case"].startswith(("walk", "baseline_walk"))]
    res["walking_cells_by_class"] = {k: int(sum(c["cls"] == k for c in walk)) for k in ("U", "M", "O")}
    res["walking_observable_with_low_visibility"] = [c for c in walk if c["cls"] == "O" and c["vis_db"] < 3.0]
    # ---- post hoc, added 2026-11-20 after the first matched results; NOT part of the declared criteria.
    # What the index responds to: device spectrum against its background AFTER the earlier study's
    # gait-harmonic removal (which uses the exact simulated gait phase), over each reference band.
    g = np.linspace(L["display_grid"]["fmin"], L["display_grid"]["fmax"], int(L["display_grid"]["n"]))
    bands = dict(zip([m for m, _ in L["references"]["table3"]["refs"]], L["references"]["table3"]["bands"]))
    raw = {(c["case"], c["planet"], c["device"], c["mode"]): c for c in cells}
    post = []
    for sid, byp in L["data"].items():
        for p, lr in byp.items():
            if "spectra" not in lr:
                continue
            for dev in ("watch", "phone"):
                S, B = np.asarray(lr["spectra"][dev]["S"], float), np.asarray(lr["spectra"][dev]["B"], float)
                for m, (a, b) in bands.items():
                    k = (g >= a) & (g <= b)
                    if k.sum() < 2:
                        k = np.abs(g - 0.5 * (a + b)) <= max(0.5 * (b - a), g[1] - g[0])
                    va = float(10 * np.log10(np.nanmean(S[k]) / np.nanmean(B[k])))
                    rc = raw.get((sid, p, dev, m))
                    post.append(dict(case=sid, planet=p, device=dev, mode=m, cls=SHORT[lr["table3"][dev]["pmode"][m]["band"]],
                                     oi=float(lr["table3"][dev]["pmode"][m]["oi"]), vis_after_removal_db=va,
                                     vis_raw_db=None if rc is None else rc["vis_db"], found=None if rc is None else rc["found"]))
    tab2 = {}
    for c in post:
        v = c["vis_after_removal_db"]
        k = "< 3 dB" if v < 3 else ("3-10 dB" if v < 10 else ">= 10 dB")
        tab2.setdefault(k, {"U": 0, "M": 0, "O": 0})[c["cls"]] += 1
    expl = [c for c in post if any(c["case"] == b["case"] and c["planet"] == b["planet"] and c["device"] == b["device"]
                                   and c["mode"] == b["mode"] for b in bad)]
    walkO = [c for c in post if c["case"].startswith(("walk", "baseline_walk")) and c["cls"] == "O"]
    res["post_hoc_summary"] = dict(
        contradiction_cells_after_removal_db=[min(c["vis_after_removal_db"] for c in expl), max(c["vis_after_removal_db"] for c in expl)] if expl else None,
        walking_observable_cells=len(walkO),
        walking_observable_min_after_removal_db=min(c["vis_after_removal_db"] for c in walkO) if walkO else None)
    res["post_hoc_after_harmonic_removal"] = dict(
        note="Added after the first matched results (2026-11-20), not part of the declared criteria: band-mean ratio of "
             "the device spectrum to its background after the earlier study's gait-harmonic removal (exact simulated gait "
             "phase), over the Table 3 reference bands, display grid.",
        cross_table=tab2, cells=post,
        contradictions_explained=expl)
    res["verdict"] = ("thresholds usable as a descriptive classification for this body model (Earth)"
                      if res["criterion1"]["holds"] and res["criterion2"]["holds"]
                      else "thresholds NOT transferable to this body model by the declared criteria; "
                           "only the threshold-free measures carry conclusions")
    json.dump(res, open(os.path.join(ROOT, "outputs", "tables", "threshold_check.json"), "w"), indent=1)
    print("criterion 1:", res["criterion1"]["holds"], [(c["case"], c["device"], c["B1_class"], round(c["B1_oi"] or 0, 1)) for c in c1])
    print("criterion 2:", res["criterion2"]["holds"], f"{len(bad)} contradictions in {len(cells)} cells")
    for c in bad:
        print("   ", c)
    print("cross-table:", json.dumps(tab))
    print("verdict:", res["verdict"])
    print("post hoc, after harmonic removal:", json.dumps(tab2))


if __name__ == "__main__":
    main()
