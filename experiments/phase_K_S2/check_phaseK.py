"""Phase K: the criteria declared in docs/phase_reports/PHASE_K_declared_rules.md, evaluated for
S2 and S2L, with S1's own results beside them.  Writes outputs/tables/phaseK_check.json.

    python experiments/phase_K_S2/check_phaseK.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T = lambda n: os.path.join(ROOT, "outputs", "tables", n)
WALK = ("walk_same_E1", "walk_same_E2", "walk3d_E1", "walk3d_E2")
DEV = {"wrist": "watch", "thigh": "phone"}


def load(n):
    return json.load(open(T(n))) if os.path.exists(T(n)) else None


def structure_summary(name, O, ST):
    alt = name != "S1"
    rk, ck = ("rec_alt", "rec_alt_chance") if alt else ("rec", "rec_chance")
    b1 = O["modes"][0]["f"]
    s = dict(structure=name, modes=O["modes"])
    # criterion 1: standing, every planet, both devices recover B1 (empty or occupied), above the chance count
    st = O["data"]["stand_E1"]
    c1 = {}
    for p, d in st.items():
        for key, dev in DEV.items():
            r = d[key]
            found = r[rk]["matched"] if alt else [q[0] for q in r["rec"]["pairs"]]
            c1[f"{p} {dev}"] = dict(B1=any(abs(x - b1) < 1e-9 for x in found), tp=r[rk]["tp"], chance=r[ck]["tp"],
                                   fp=r[rk]["fp"], tp_empty_only=r["rec"]["tp"])
    s["c1_standing"] = dict(cells=c1, holds=all(v["B1"] and v["tp"] > v["chance"] for v in c1.values()))
    # criterion 2: walking, blind recovery without removal at most chance + 1 in every case
    c2 = {}
    for sid in WALK:
        for p, d in O["data"].get(sid, {}).items():
            for key, dev in DEV.items():
                r = d[key]
                c2[f"{sid} {p} {dev}"] = dict(tp=r[rk]["tp"], chance=r[ck]["tp"], fp=r[rk]["fp"])
    s["c2_walking"] = dict(cells=c2, n=len(c2), above=[k for k, v in c2.items() if v["tp"] > v["chance"] + 1],
                           holds_strict=all(v["tp"] <= v["chance"] + 1 for v in c2.values()),
                           holds=sum(v["tp"] > v["chance"] + 1 for v in c2.values()) <= 1)   # amended, like-for-like with S1
    # criterion 3: order-tracked removal on the device's own clock: at least one case above chance + 1
    if ST:
        c3 = {}
        for sid, byp in ST["data"].items():
            if sid not in WALK:
                continue
            for p, r in byp.items():
                for dev in ("watch", "phone"):
                    m2 = r["m2"]
                    cell = {}
                    for v in ("none", "device", "exact"):
                        if v in m2 and dev in m2[v]:
                            q = m2[v][dev]["no_rejection"]
                            cell[v] = dict(tp=q["tp"], chance=q["chance_tp"], fp=q["fp"])
                    c3[f"{sid} {p} {dev}"] = cell
        cnt = {v: sum(1 for c in c3.values() if v in c and c[v]["tp"] > c[v]["chance"] + 1) for v in ("none", "device", "exact")}
        s["c3_device_clock"] = dict(cells=c3, n=len(c3), above_chance_plus_one=cnt, holds=cnt["device"] >= 1)
    # criterion 4: occupied B1 from the standing trials against the linear prediction (within 2 %)
    if alt:
        c4 = {p: dict(trials=d.get("occupied_B1_from_trials"), predicted=d.get("occupied_B1_predicted"),
                      diff_rel=(d["occupied_B1_from_trials"] / d["occupied_B1_predicted"] - 1)
                      if d.get("occupied_B1_from_trials") else None) for p, d in st.items()}
        s["c4_model"] = dict(cells=c4, holds=all(v["diff_rel"] is not None and abs(v["diff_rel"]) <= 0.02 for v in c4.values()))
    s["deck_rms_under"] = {sid: {p: d["deck_rms_under"]["mean"] for p, d in O["data"][sid].items()}
                           for sid in ("stand_E1",) + WALK if sid in O["data"]}
    s["falls"] = {sid: {p: d.get("falls", 0) for p, d in O["data"][sid].items()} for sid in WALK if sid in O["data"]}
    return s


def main():
    out = dict(declared="docs/phase_reports/PHASE_K_declared_rules.md", structures={})
    for name, on, sn in (("S1", "outcomes.json", "step_timing.json"), ("S2", "outcomes_S2.json", "step_timing_S2.json"),
                         ("S2L", "outcomes_S2L.json", "step_timing_S2L.json")):
        O = load(on)
        if O is None:
            continue
        out["structures"][name] = structure_summary(name, O, load(sn))
        s = out["structures"][name]
        print(f"{name}: C1 standing {s['c1_standing']['holds']} | C2 walking hidden {s['c2_walking']['holds']} "
              f"({len(s['c2_walking']['above'])} of {s['c2_walking']['n']} above chance+1) | "
              f"C3 {s.get('c3_device_clock', {}).get('above_chance_plus_one')} | C4 {s.get('c4_model', {}).get('holds')}")
    json.dump(out, open(T("phaseK_check.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
