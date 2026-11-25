"""Stage 1 summary: 3-D walker against the side-view walker (Earth, 1.0 m/s).

Reads outputs/gait/earth_1.00.json, outputs/gait/earth3d_1.00.json, the trial
records and outputs/tables/outcomes.json; applies the decision rule declared
in docs/phase_reports/PHASE_G_walker3d.md before the 3-D trials were run;
writes outputs/tables/stage1_summary.json and prints a Markdown digest.

    python experiments/phase_G_walk3d/summarize_stage1.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GAIT = os.path.join(ROOT, "outputs", "gait")
REC = os.path.join(ROOT, "outputs", "raw", "records")
SENS = {"wrist": "wrist", "thigh": "thigh pocket"}
PAIRS = [("walk_same_E1", "walk3d_E1", "shakers on"), ("walk_same_E2", "walk3d_E2", "footsteps only")]
GAIT_KEYS = ["speed", "step_frequency", "step_length", "duty_factor", "double_support_fraction", "dlf1",
             "grf_foot_peak_over_W", "grf_peak_over_W", "com_vertical_excursion", "swing_scuffs_per_step",
             "step_width", "grf_lateral_peak_over_W", "dlf_lat", "com_lateral_excursion", "heading_max", "lateral_drift"]


def trial_stats(cond):
    from planetsense.simulation.records import analysis_window, load_record
    out = {}
    for exc in ("E1", "E2", "rigid"):
        ps = sorted(glob.glob(os.path.join(REC, f"walking_{cond}_{exc}_r*.npz")))
        rs = [load_record(p) for p in ps]
        if not rs:
            continue
        d = dict(n=len(rs), falls=int(sum(bool(r["fell"]) for r in rs)),
                 energy_drift_max=float(max(float(r["energy_drift_rel"]) for r in rs)))
        for k in ("speed", "step_frequency", "step_width", "grf_lateral_peak_over_W", "com_lateral_excursion"):
            v = [r["gait"].get(k) for r in rs if r["gait"].get(k) is not None]
            if v:
                d[k] = [float(np.mean(v)), float(np.std(v))]
        if exc != "rigid":
            rms = []
            for r in rs:
                if "a_under" in r:
                    m = analysis_window(r)                    # same window as the outcome analysis
                    if m.any():
                        rms.append(float(np.asarray(r["a_under"])[m, 2].std()))
            if rms:
                d["deck_rms_under"] = [float(np.mean(rms)), float(np.std(rms))]
        # lateral path on the deck (centre of mass)
        ys = []
        for r in rs:
            y = np.asarray(r["com"])[:, 1]
            m = analysis_window(r)
            if m.any():
                ys.append([float(y[m].min()), float(y[m].max())])
        if ys:
            d["com_y_range_on_deck"] = [min(a for a, _ in ys), max(b for _, b in ys)]
        out[exc] = d
    return out


def main():
    g2 = json.load(open(os.path.join(GAIT, "earth_1.00.json")))
    g3 = json.load(open(os.path.join(GAIT, "earth3d_1.00.json")))
    m2, m3 = g2.get("metrics_validation_30s", {}), g3.get("metrics_validation_30s", {})
    O = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json")))
    D, modes = O["data"], [m["label"] for m in O["modes"]]
    res = dict(gait=dict(side_view={k: m2.get(k) for k in GAIT_KEYS}, three_d={k: m3.get(k) for k in GAIT_KEYS},
                         targets_3d=g3.get("targets"), params_3d=g3.get("params"), fixed_3d=g3.get("fixed"),
                         final_selection=g3.get("final_selection"), generations=g3.get("generations")),
               trials=dict(side_view=trial_stats("earth_1.00"), three_d=trial_stats("earth3d_1.00")),
               modes=modes, comparison=[])
    # gait background: how much each walker's own movement puts into the worn sensors in each bridge
    # band (noisy rigid-ground twins, same estimator as the visibility analysis)
    import sys
    sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_F_outcomes"))
    import outcomes as oc
    f_modes = [m["f"] for m in O["modes"]]
    bg = {}
    for who, cond in (("side_view", "earth_1.00"), ("three_d", "earth3d_1.00")):
        recs = oc.load_group("walking", "rigid", "Earth", cond)
        f, S = oc.spectra_of(recs, 4.0, True)
        bg[who] = {sk: [float(np.mean(S[sk][oc.band(f, f0)])) for f0 in f_modes] for sk in SENS}
    res["background_3d_minus_side_db"] = {sk: [float(10 * np.log10(a / b)) for a, b in zip(bg["three_d"][sk], bg["side_view"][sk])]
                                          for sk in SENS}
    # per-axis spectra of the noise-free sensors on rigid ground (sensor frame: x forward, y sideways, z up),
    # dB re 1 (m/s^2)^2/Hz, band means; shows where each walker's own movement is weak or strong
    from scipy.signal import welch
    from planetsense.simulation.records import load_record
    axes = {}
    for who, cond in (("side_view", "earth_1.00"), ("three_d", "earth3d_1.00")):
        axes[who] = {}
        for sk, sns in oc.SENSORS.items():
            Ps = []
            for pth in sorted(glob.glob(os.path.join(REC, f"walking_{cond}_rigid_r*.npz"))):
                r = load_record(pth)
                for sn in sns:                                       # left and right
                    X = np.asarray(r["f"], float)[oc.window(r)][:, list(r["sensors"]).index(sn), :]
                    ff, P = welch(X - X.mean(0), fs=float(r["fs"]), nperseg=int(4 * float(r["fs"])), axis=0)
                    Ps.append(P)
            P = np.mean(Ps, axis=0)
            axes[who][sk] = [[float(10 * np.log10(v)) if v > 0 else None for v in P[oc.band(ff, f0)].mean(0)] for f0 in f_modes]
    res["sensor_axes_rigid_db"] = axes
    holds = True
    for s2, s3, lab in PAIRS:
        for sk in SENS:
            d2, d3 = D[s2]["Earth"][sk], D[s3]["Earth"][sk]
            ok3 = d3["rec"]["tp"] <= d3["rec_chance"]["tp"] + 1
            holds &= ok3
            res["comparison"].append(dict(excitation=lab, sensor=SENS[sk],
                                          vis_side=d2["vis_db"], vis_3d=d3["vis_db"],
                                          rec_side=d2["rec"], chance_side=d2["rec_chance"],
                                          rec_3d=d3["rec"], chance_3d=d3["rec_chance"], rule_met_3d=bool(ok3)))
    res["decision"] = dict(rule="3-D: modes found blind <= modes found by the same rule on the rigid-ground twin + 1, "
                                "for both sensors and both excitations (declared before the 3-D trials)",
                           holds=bool(holds))
    # Stage 2 (2026-11-20): the same comparison and rule on every planet that has a 3-D walker
    res["by_planet"] = {}
    for planet, c2, c3 in (("Earth", "earth_1.00", "earth3d_1.00"), ("Mars", "mars_1.00", "mars3d_1.00"),
                           ("Moon", "moon_1.00", "moon3d_1.00")):
        if planet not in D.get("walk3d_E1", {}) or not os.path.exists(os.path.join(GAIT, c3 + ".json")):
            continue
        g3p = json.load(open(os.path.join(GAIT, c3 + ".json")))
        g2p = json.load(open(os.path.join(GAIT, c2 + ".json")))
        comp, ok_all = [], True
        for s2, s3, lab in PAIRS:
            for sk in SENS:
                d2, d3 = D[s2][planet][sk], D[s3][planet][sk]
                ok3 = d3["rec"]["tp"] <= d3["rec_chance"]["tp"] + 1
                ok_all &= ok3
                comp.append(dict(excitation=lab, sensor=SENS[sk], vis_side=d2["vis_db"], vis_3d=d3["vis_db"],
                                 rec_side=d2["rec"], chance_side=d2["rec_chance"], rec_3d=d3["rec"],
                                 chance_3d=d3["rec_chance"], rule_met_3d=bool(ok3)))
        res["by_planet"][planet] = dict(
            gait=dict(side_view={k: g2p.get("metrics_validation_30s", {}).get(k) for k in GAIT_KEYS},
                      three_d={k: g3p.get("metrics_validation_30s", {}).get(k) for k in GAIT_KEYS},
                      targets_3d=g3p.get("targets"), final_selection=g3p.get("final_selection"),
                      generations=g3p.get("generations")),
            trials=dict(side_view=trial_stats(c2), three_d=trial_stats(c3)), comparison=comp, holds=bool(ok_all))
    res["decision_by_planet"] = {p_: v["holds"] for p_, v in res["by_planet"].items()}
    json.dump(res, open(os.path.join(ROOT, "outputs", "tables", "stage1_summary.json"), "w"), indent=1)
    print("decision by planet:", res["decision_by_planet"])

    f = lambda v, n=3: "–" if v is None else (f"{v:.{n}g}" if isinstance(v, float) else str(v))
    print("| Descriptor | side view | 3-D | 3-D target |\n|---|---|---|---|")
    tg = g3.get("targets") or {}
    for k in GAIT_KEYS:
        t = tg.get(k)
        print(f"| {k} | {f(m2.get(k))} | {f(m3.get(k))} | {'' if t is None else ('≤ ' if t[2] == 'max' else '') + f'{t[0]:g} ± {t[1]:g}'} |")
    print()
    for c in res["comparison"]:
        print(f"- {c['excitation']}, {c['sensor']}: visibility side view " + ", ".join(f"{m} {v:.1f}" for m, v in zip(modes, c["vis_side"]))
              + "; 3-D " + ", ".join(f"{m} {v:.1f}" for m, v in zip(modes, c["vis_3d"]))
              + f". Recovered side view {c['rec_side']['tp']} (+{c['rec_side']['fp']} false; chance {c['chance_side']['tp']}), "
              f"3-D {c['rec_3d']['tp']} (+{c['rec_3d']['fp']} false; chance {c['chance_3d']['tp']}); rule met: {c['rule_met_3d']}")
    print()
    for sk in SENS:
        print(f"- gait background, {SENS[sk]}: 3-D minus side view (dB) " +
              ", ".join(f"{m} {v:+.1f}" for m, v in zip(modes, res["background_3d_minus_side_db"][sk])))
    print()
    for sk in SENS:
        for who in ("side_view", "three_d"):
            print(f"- rigid-ground {SENS[sk]} axes (x, y, z; dB re 1 (m/s²)²/Hz), {who}: " + "; ".join(
                f"{m} " + "/".join("–" if v is None else f"{v:.1f}" for v in vals) for m, vals in zip(modes, res["sensor_axes_rigid_db"][who][sk])))
    print()
    for who in ("side_view", "three_d"):
        print(who, json.dumps(res["trials"][who]))
    print("\nDECISION: walking conclusion", "HOLDS" if holds else "DOES NOT HOLD", "for the 3-D walker")


if __name__ == "__main__":
    main()
