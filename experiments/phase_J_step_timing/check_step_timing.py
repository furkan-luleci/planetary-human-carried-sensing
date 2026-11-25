"""Evaluate the criteria declared in docs/phase_reports/PHASE_J_step_timing.md
(commit f784e2d, before the device-clock results were computed).

1. Index classes robust to the device's own timing: (a) >= 90 % of the 266 walking
   mode cells keep their class; (b) >= 90 % of the cells observable with the exact
   clock stay observable.
2. Phase I criterion 1 re-checked with the device clock: B1 observable for both
   devices on the 1.5 m/s baseline walk (reported).
3. Walking conclusion with order-tracked removal on the device's own clock (M2,
   harmonic rejection off): in each of the 38 walking cells, modes recovered blind
   <= chance + 1.

Also descriptive: band excess after removal (device minus exact clock), M2 with the
exact clock and without removal, the secondary rule (rejection on), and the per-side
regression of the exact path against Phase I.

Reads outputs/tables/step_timing.json; writes outputs/tables/step_timing_check.json.

    python experiments/phase_J_step_timing/check_step_timing.py
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEVS = ("watch", "phone")


def main():
    J = json.load(open(os.path.join(ROOT, "outputs", "tables", "step_timing.json")))
    modes = [m for m, _ in J["references"]["refs"]]
    cells, m2 = [], []
    reg = []
    for sid, byp in J["data"].items():
        for p, r in byp.items():
            reg.append(dict(case=sid, planet=p, **r["regression_vs_phase_I"]))
            for dev in DEVS:
                ex, dv = r["exact"][dev], r["device"][dev]
                for m in modes:
                    cells.append(dict(case=sid, planet=p, device=dev, mode=m, exact=ex["pmode"][m]["cls"],
                                      device_cls=dv["pmode"][m]["cls"], oi_exact=ex["pmode"][m]["oi"],
                                      oi_device=dv["pmode"][m]["oi"], excess_exact=ex["excess_db"][m],
                                      excess_device=dv["excess_db"][m]))
                row = dict(case=sid, planet=p, device=dev)
                for var in ("none", "exact", "device"):
                    for rule in ("no_rejection", "rejection"):
                        d = r["m2"][var][dev][rule]
                        row[f"{var}_{rule}"] = dict(tp=d["tp"], fp=d["fp"], chance=d["chance_tp"], found=d["found"])
                m2.append(row)
    n = len(cells)
    same = sum(c["exact"] == c["device_cls"] for c in cells)
    o_ex = [c for c in cells if c["exact"] == "O"]
    o_kept = sum(c["device_cls"] == "O" for c in o_ex)
    c1 = dict(cells=n, same_class=same, share_same=same / n, observable_exact=len(o_ex), observable_kept=o_kept,
              share_kept=(o_kept / len(o_ex)) if o_ex else None)
    c1["holds"] = bool(c1["share_same"] >= 0.9 and (c1["share_kept"] is None or c1["share_kept"] >= 0.9))
    trans = {}
    for c in cells:
        k = f"{c['exact']}->{c['device_cls']}"
        trans[k] = trans.get(k, 0) + 1
    c1["transitions"] = trans
    c1["changed"] = [c for c in cells if c["exact"] != c["device_cls"]]
    b1 = [c for c in cells if c["case"] == "baseline_walk" and c["mode"] == "B1"]
    c2 = dict(cells=[dict(device=c["device"], exact=c["exact"], device_clock=c["device_cls"],
                          oi_exact=c["oi_exact"], oi_device=c["oi_device"]) for c in b1],
              holds=bool(b1) and all(c["device_cls"] == "O" for c in b1))
    over = [r for r in m2 if r["device_no_rejection"]["tp"] > r["device_no_rejection"]["chance"] + 1]
    c3 = dict(cells=len(m2), above_chance_plus_one=[dict(case=r["case"], planet=r["planet"], device=r["device"],
                                                         **r["device_no_rejection"]) for r in over],
              holds=len(over) == 0)
    desc = {}
    for var in ("none", "exact", "device"):
        for rule in ("no_rejection", "rejection"):
            k = f"{var}_{rule}"
            desc[k] = dict(cells_above_chance_plus_one=sum(r[k]["tp"] > r[k]["chance"] + 1 for r in m2),
                           total_found=sum(r[k]["tp"] for r in m2), total_chance=sum(r[k]["chance"] for r in m2))
    dx = np.array([c["excess_device"] - c["excess_exact"] for c in cells])
    by_mode = {m: float(np.median([c["excess_device"] - c["excess_exact"] for c in cells if c["mode"] == m])) for m in modes}
    res = dict(declared="docs/phase_reports/PHASE_J_step_timing.md (commit f784e2d)",
               criterion1=c1, criterion2_phase_I_c1_recheck=c2, criterion3=c3, m2_summary=desc, m2_cells=m2,
               excess_change_db=dict(median=float(np.median(dx)), p10=float(np.percentile(dx, 10)),
                                     p90=float(np.percentile(dx, 90)), by_mode_median=by_mode),
               regression_vs_phase_I=dict(max_abs_oi_difference=max(x["max_abs_oi_difference"] for x in reg),
                                          class_mismatches=sum(x["class_mismatches"] for x in reg)),
               cells=cells)
    res["verdict"] = dict(
        index=("walking classes robust to the device's own gait timing" if c1["holds"] else
               "walking classes depend on gait timing more precise than the device's own"),
        walking=("walking conclusion holds with order-tracked removal on the device's own clock" if c3["holds"] else
                 "walking conclusion qualified: with order-tracked removal on the device's own clock, recovery exceeds "
                 "chance + 1 in the listed cells"))
    json.dump(res, open(os.path.join(ROOT, "outputs", "tables", "step_timing_check.json"), "w"), indent=1)
    print(f"regression vs Phase I: max |dOI| {res['regression_vs_phase_I']['max_abs_oi_difference']:.3g}, "
          f"class mismatches {res['regression_vs_phase_I']['class_mismatches']}")
    print(f"criterion 1: {c1['holds']}  same class {same}/{n} ({100 * c1['share_same']:.0f} %), observable kept "
          f"{o_kept}/{len(o_ex)}" + (f" ({100 * c1['share_kept']:.0f} %)" if o_ex else "") + f"  transitions {trans}")
    print(f"criterion 2 (recheck of Phase I c1 with device clock): {c2['holds']}  {c2['cells']}")
    print(f"criterion 3: {c3['holds']}  cells above chance+1: {len(over)} of {len(m2)}")
    for r in over:
        print("   ", r["case"], r["planet"], r["device"], r["device_no_rejection"])
    print("M2 summary:", json.dumps(desc))
    print(f"excess change (device - exact clock), dB: median {res['excess_change_db']['median']:.1f}, 10-90 % "
          f"{res['excess_change_db']['p10']:.1f} to {res['excess_change_db']['p90']:.1f}; by mode {by_mode}")
    print("verdict:", res["verdict"])


if __name__ == "__main__":
    main()
