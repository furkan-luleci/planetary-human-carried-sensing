"""Phase K: blind recovery after order-tracked removal of the gait's harmonics (Phase J, M2) on
S2 or S2L, exactly as on S1 (experiments/phase_J_step_timing/step_timing_test.py): the vendored
removal with no removal, the exact clock and each device's own clock; S1's frozen recovery rule
with and without harmonic rejection; truth = the structure's modes below 30 Hz, counted at the
empty or the declared occupied frequency (and at the empty frequency only); chance = the same
on S1's rigid-ground twins cut to the structure's crossing.  The earlier study's index is not
computed: its references and thresholds belong to S1.

Writes outputs/tables/step_timing_<S>.json.

    PYTHONPATH=src python experiments/phase_K_S2/step_timing_s2.py S2 [--workers 4]
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_J_step_timing"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_I_baseline_match"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step_timing_test as STT  # noqa: E402

MAIN = {"walk_same_E1": ("E1", {"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}),
        "walk_same_E2": ("E2", {"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}),
        "walk3d_E1": ("E1", {"Earth": "earth3d_1.00", "Mars": "mars3d_1.00", "Moon": "moon3d_1.00"}),
        "walk3d_E2": ("E2", {"Earth": "earth3d_1.00", "Mars": "mars3d_1.00", "Moon": "moon3d_1.00"}),
        # Phase O (2026-11-22): the what-if gaits, now simulated on the habitat floor too, so that both structures
        # have the same cases.  The declared criterion C3 (check_phaseK.py) still counts the four situations above only.
        "walk_similar_E1": ("E1", {"Earth": "earth_1.25", "Mars": "mars_0.77_similar", "Moon": "moon_0.51_similar"}),
        "walk_similar_E2": ("E2", {"Earth": "earth_1.25", "Mars": "mars_0.77_similar", "Moon": "moon_0.51_similar"})}


def _rep(p):
    return int(re.search(r"_r(\d+)\.npz$", p).group(1))


def tasks(struct):
    rec_dir = os.path.join(ROOT, "outputs", "raw", "records_" + struct)
    twin_dir = os.path.join(ROOT, "outputs", "raw", "records")
    out = []
    for sid, (exc, conds) in MAIN.items():
        for planet, cond in conds.items():
            fs = sorted(glob.glob(os.path.join(rec_dir, f"walking_{cond}_{exc}_r*.npz")), key=_rep)
            tw = sorted(glob.glob(os.path.join(twin_dir, f"walking_{cond}_rigid_r*.npz")), key=_rep)
            common = sorted(set(map(_rep, fs)) & set(map(_rep, tw)))
            if common:
                out.append(dict(struct=struct, sid=sid, planet=planet, cond=cond,
                                files=[f for f in fs if _rep(f) in common], twins=[f for f in tw if _rep(f) in common]))
    return out


def snap(found, modes, occ_p):
    out = []
    for x in found:
        y = x
        for m in modes:
            fo = occ_p.get(m["label"], m["f"])
            if abs(x - fo) <= max(0.15, 0.03 * fo) and abs(x - m["f"]) > max(0.15, 0.03 * m["f"]):
                y = m["f"]
                break
        out.append(y)
    return out


def m2_recover2(f, S, Bs, rule, modes, occ_p, f_stride):
    from planetsense.identification.spectral import match_peaks, pick_peaks
    truth = [m["f"] for m in modes]
    out = {}
    for name, fh in (("no_rejection", None), ("rejection", f_stride)):
        pk = [[q[0] for q in pick_peaks(f, spec, fmin=1.0, fmax=30.0, prominence_db=rule["prom"], background_hz=rule["bg"],
                                       harmonics_of=fh, harmonic_halfwidth=rule["hw"], max_harmonic_hz=rule["hmax"])]
              for spec in (S, Bs)]
        e = [match_peaks(x, truth, tol_rel=0.03, tol_abs=0.15) for x in pk]
        a = [match_peaks(snap(x, modes, occ_p), truth, tol_rel=0.03, tol_abs=0.15) for x in pk]
        out[name] = dict(tp=int(a[0]["tp"]), fp=int(a[0]["fp"]), chance_tp=int(a[1]["tp"]), chance_fp=int(a[1]["fp"]),
                         matched=[float(q[0]) for q in a[0]["pairs"]],
                         empty_only=dict(tp=int(e[0]["tp"]), fp=int(e[0]["fp"]), chance_tp=int(e[1]["tp"]),
                                         chance_fp=int(e[1]["fp"])))
    return out


def run_task(tk):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import legacy_pipeline as LP
    from planetsense.evaluation import baseline_match as MM
    from planetsense.simulation.records import load_record
    from planetsense.structure import registry as SR
    t0 = time.time()
    md = SR.modal_deck(tk["struct"])
    L = float(md.params.L)
    modes = [dict(label=l, f=float(q)) for l, q in zip(md.labels, md.freq_hz) if q < 30.0]
    occ = json.load(open(os.path.join(ROOT, "outputs", "tables", "occupied_modes.json")))["data"][tk["struct"]]
    occ_p = {l: (v["f"] if v else occ["empty"][l]) for l, v in occ["occupied"][tk["planet"]].items()}
    rule = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json")))["recovery_rule"]["settings"]
    recs = [dict(load_record(f), deck_L=L) for f in tk["files"]]
    twins = [dict(load_record(f), deck_L=L) for f in tk["twins"]]
    keep = [i for i in range(len(recs)) if not recs[i].get("fell") and not twins[i].get("fell")]
    recs, twins = [recs[i] for i in keep], [twins[i] for i in keep]
    mv = json.load(open(os.path.join(ROOT, "outputs", "gait", tk["cond"] + ".json")))["metrics_validation_30s"]
    fp_nom = float(mv["step_frequency"])
    clocks = STT.device_clocks(recs, twins)
    res = dict(struct=tk["struct"], sid=tk["sid"], planet=tk["planet"], cond=tk["cond"], n_records=len(recs),
               falls=len(tk["files"]) - len(keep), fp_nominal=fp_nom, rule=rule, modes=modes, occupied=occ_p, m2={})
    for variant in ("exact", "device"):
        sides = {}
        for dev, pairs in STT.SENSORS.items():
            sides[dev] = []
            for ours, base in pairs:
                if variant == "exact":
                    E = MM.ensemble(recs, False, fp=fp_nom, sensors=[base])
                    B = MM.ensemble(twins, False, fp=fp_nom, sensors=[base])
                else:
                    c = clocks[ours]
                    E = MM.ensemble(recs, False, fp=c["fp"], sensors=[base], clocks=c["clocks_rec"])
                    B = MM.ensemble(twins, False, fp=c["fp"], sensors=[base], clocks=c["clocks_twin"])
                sides[dev].append((E, B))
        with STT._Capture(LP) as cap:
            MM.device_curves_sided(sides)
        names = ("none", variant) if variant == "exact" else (variant,)
        for dev, pairs in STT.SENSORS.items():
            fstr = float(np.mean([1.0 / clocks[o]["stride_s"] for o, _b in pairs]))
            for vname in names:
                cf, cb = [], []
                for (E, B), (_o, base) in zip(sides[dev], pairs):
                    if vname == "none":
                        cf.append(STT.raw_channels(LP, E, base)); cb.append(STT.raw_channels(LP, B, base))
                    else:
                        cf.append([cap.store[(id(E), i, base)] for i in range(len(E["passes"]))])
                        cb.append([cap.store[(id(B), i, base)] for i in range(len(B["passes"]))])
                f, S = STT.m2_spectra(cf, MM.FS_IMU)
                _, Bs = STT.m2_spectra(cb, MM.FS_IMU)
                k = f <= 30.0
                res["m2"].setdefault(vname, {})[dev] = dict(m2_recover2(f, S, Bs, rule, modes, occ_p, fstr), f_stride=fstr,
                                                            f=[round(float(v), 4) for v in f[k]],
                                                            S=[float(f"{v:.4g}") for v in S[k]],
                                                            B=[float(f"{v:.4g}") for v in Bs[k]])
    res["timing"] = {s: dict(n_e=c["n_e"], stride_s=c["stride_s"],
                             random_ms_median=float(np.median([a["random_ms"] for a in c["accuracy"]])),
                             total_ms_median=float(np.median([a["total_ms"] for a in c["accuracy"]])))
                     for s, c in clocks.items()}
    res["wall"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("struct", choices=("S2", "S2L"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tks = tasks(a.struct)
    print(len(tks), "walking cases on", a.struct, flush=True)
    with Pool(a.workers) as pool:
        results = list(pool.imap(run_task, tks))
    out = dict(structure=a.struct, declared="docs/phase_reports/PHASE_K_declared_rules.md", noise_ug=STT.NOISE_UG, data={})
    for r in results:
        out["data"].setdefault(r["sid"], {})[r["planet"]] = r
        line = f"{r['sid']:13s} {r['planet']:5s} n {r['n_records']:2d} |"
        for dev in STT.SENSORS:
            line += f" {dev}: " + " ".join(f"{v}:{r['m2'][v][dev]['no_rejection']['tp']}/{r['m2'][v][dev]['no_rejection']['chance_tp']}"
                                           for v in ("none", "device", "exact")) + " |"
        print(line, flush=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "tables", f"step_timing_{a.struct}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
