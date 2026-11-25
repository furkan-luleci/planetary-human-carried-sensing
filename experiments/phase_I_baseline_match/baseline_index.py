"""The Observability Index of the earlier study on this study's trials, computed with
the baseline's own processing (src/planetsense/evaluation/baseline_match.py).

For every situation, planet and device (watch = left and right wrist; phone =
left and right trouser pocket):
  * the 20 records and their matched background (walking: the rigid-ground
    twins; standing: sensor noise alone), with the same sensor-noise
    realization in each record and its background (75 ug/sqrt(Hz));
  * reference frequencies and half-power bands of the earlier study's Table 3
    (the deck here is the same deck: identical finite-element frequencies and
    damping; experiments/phase_I_baseline_match/ambient_test.py repeats the
    ambient test on this project's deck model and is used as a sensitivity);
  * classes with the calibrated thresholds (27.5, 92.3) and with both
    thresholds lowered or raised by 25 % (the raised upper threshold is capped
    at 100, the highest possible score).

Also the earlier study's baseline, as a check of the thresholds on this body
model (Earth only): standing at mid-span, and walking at 1.5 m/s.

Writes outputs/tables/legacy_oi_baseline.json and a digest, outputs/tables/legacy_oi_baseline.md.

    python experiments/phase_I_baseline_match/baseline_index.py [--workers 4]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REC = os.environ.get("PLANETSENSE_REC_DIR", os.path.join(ROOT, "outputs", "raw", "records"))
GAIT = os.path.join(ROOT, "outputs", "gait")
PLANETS = ["Earth", "Mars", "Moon"]
T1, T2 = 27.5, 92.3
VARIANTS = {"nominal": (T1, T2), "lower25": (0.75 * T1, 0.75 * T2), "higher25": (1.25 * T1, min(1.25 * T2, 100.0))}
# the earlier study, Table 3, baseline deck: (label, reference frequency), half-power band
TABLE3_REFS = [("B1", 1.8024), ("B2", 7.2101), ("T1", 12.2911), ("B3", 16.1523), ("L1", 22.4389), ("T2", 24.8663),
               ("T3", 37.2076)]
TABLE3_BANDS = [(1.7650, 1.8329), (7.1281, 7.3358), (11.9098, 12.6221), (15.7645, 16.5609), (22.0076, 22.6826),
                (24.3934, 25.1996), (36.9920, 38.0416)]
SCENARIOS = [
    dict(id="stand_E1", label="Standing at 4.2 m, shakers on", kind="standing", pat="standing_{p}_E1_r*.npz",
         planets=PLANETS),
    dict(id="walk_same_E1", label="Walking 1.0 m/s, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}),
    dict(id="walk_similar_E1", label="Earth gait slowed down, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth_1.25", "Mars": "mars_0.77_similar", "Moon": "moon_0.51_similar"}),
    dict(id="walk_same_E2", label="Walking 1.0 m/s, footsteps only", kind="walking", exc="E2",
         conds={"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}),
    dict(id="walk_similar_E2", label="Earth gait slowed down, footsteps only", kind="walking", exc="E2",
         conds={"Earth": "earth_1.25", "Mars": "mars_0.77_similar", "Moon": "moon_0.51_similar"}),
    dict(id="walk3d_E1", label="3-D walker 1.0 m/s, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth3d_1.00", "Mars": "mars3d_1.00", "Moon": "moon3d_1.00"}),
    dict(id="walk3d_E2", label="3-D walker 1.0 m/s, footsteps only", kind="walking", exc="E2",
         conds={"Earth": "earth3d_1.00", "Mars": "mars3d_1.00", "Moon": "moon3d_1.00"}),
    # the earlier study's baseline (threshold check): mid-span standing; walking at 1.5 m/s
    dict(id="baseline_stand", label="earlier-study baseline: standing at mid-span, shakers on", kind="standing",
         pat="standing_Earth-mid_E1_r*.npz", planets=["Earth"]),
    dict(id="baseline_walk", label="earlier-study baseline: walking 1.5 m/s, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth_1.50"}),
]
# the earlier study's baseline results (Section 3.2), for the check
EARLIER_BASELINE = {
    "walking": {"watch": dict(B1=99.75, B2=97.77, T1=95.35, T3=23.48, device=70.07,
                              classes=dict(B1="O", B2="O", T1="O", B3="M", L1="M", T2="M", T3="U")),
                "phone": dict(B1=99.57, T3=60.60, device=68.76,
                              classes=dict(B1="O", B2="M", T1="M", B3="M", L1="M", T2="M", T3="M"))},
    "standing": {"watch": dict(B2=29.16, device=57.76,
                               classes=dict(B1="O", B2="M", T1="O", B3="M", L1="M", T2="M", T3="M")),
                 "phone": dict(B2=17.87, device=59.11,
                               classes=dict(B1="O", B2="U", T1="O", B3="M", L1="M", T2="M", T3="M"))},
}


def _rep(path):
    return int(re.search(r"_r(\d+)\.npz$", path).group(1))


def files(pattern):
    return sorted(glob.glob(os.path.join(REC, pattern)), key=_rep)


def tasks():
    out = []
    for sc in SCENARIOS:
        for p in (sc.get("planets") or list(sc["conds"])):
            if sc["kind"] == "standing":
                fs = files(sc["pat"].format(p=p))
                tw = None
            else:
                cond = sc["conds"][p]
                fs = files(f"walking_{cond}_{sc['exc']}_r*.npz")
                tw = files(f"walking_{cond}_rigid_r*.npz")
                common = sorted(set(map(_rep, fs)) & set(map(_rep, tw)))
                fs = [f for f in fs if _rep(f) in common]
                tw = [f for f in tw if _rep(f) in common]
            if fs:
                out.append(dict(sid=sc["id"], planet=p, kind=sc["kind"], files=fs, twins=tw,
                                cond=None if sc["kind"] == "standing" else sc["conds"][p]))
    return out


def run_task(tk):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import baseline_match as MM
    from planetsense.simulation.records import load_record
    t0 = time.time()
    recs = [load_record(f) for f in tk["files"]]
    standing = tk["kind"] == "standing"
    twins = None if standing else [load_record(f) for f in tk["twins"]]
    # a record whose walker (or whose twin's walker) fell is left out together with its pair
    keep = [i for i, r in enumerate(recs) if not r.get("fell") and (standing or not twins[i].get("fell"))]
    fell = [recs[i]["name"] for i in range(len(recs)) if i not in keep]
    recs = [recs[i] for i in keep]
    if standing:
        fp, speed = None, None
        E = MM.ensemble(recs, standing=True)
        B = MM.ensemble(recs, standing=True, zero=True)
    else:
        mv = json.load(open(os.path.join(GAIT, tk["cond"] + ".json")))["metrics_validation_30s"]
        fp, speed = float(mv["step_frequency"]), float(mv["speed"])
        twins = [twins[i] for i in keep]
        E = MM.ensemble(recs, standing=False, fp=fp)
        B = MM.ensemble(twins, standing=False, fp=fp)
    curves = MM.device_curves(E, B)
    Trec, Tres = MM.record_window_s(standing, speed)
    own = json.load(open(os.path.join(ROOT, "outputs", "tables", "reference_modes.json")))
    res = dict(sid=tk["sid"], planet=tk["planet"], n_records=len(recs), excluded_fell=fell, fp=fp, speed=speed,
               Trec=Trec, Tres=Tres, h=max(1.0 / Tres, (50.0 - 0.3) / 749), wall=None)
    for refset, refs, bands in (("table3", TABLE3_REFS, TABLE3_BANDS),
                                ("ambient_test", [tuple(x) for x in own["refs"]], [tuple(b) for b in own["bands"]])):
        r = MM.index_run(curves, refs, bands, Tres)
        res[refset] = {}
        for dev in ("watch", "phone"):
            d = r[dev]
            pm = {m: dict(oi=float(v["oi"]), band=v["band"], f=v["f"], fh=v["fh"], G=v["G"], F=v["F"],
                          Po=v["Po"], Pb=v["Pb"], W=v["W"], bnd=v["bnd"]) for m, v in d["pmode"].items()}
            res[refset][dev] = dict(oi=float(d["oi"]), band=d["band"], pmode=pm,
                                    classes={k: MM.classes(d["pmode"], *tv) for k, tv in VARIANTS.items()},
                                    n_comb_lines=curves[dev]["n_comb_lines"])
    spec = {}
    for dev in ("watch", "phone"):
        g, S, Bc = MM.display(curves[dev])
        spec[dev] = dict(S=[float(f"{v:.4g}") for v in S], B=[float(f"{v:.4g}") for v in Bc])
    res["spectra"] = spec
    res["wall"] = time.time() - t0
    return res


def short(c):
    return {"observable": "O", "marginally observable": "M", "unobservable": "U"}.get(c, "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    tk = tasks()
    print(len(tk), "conditions", flush=True)
    with Pool(a.workers) as pool:
        results = list(pool.imap(run_task, tk))
    out = dict(status="SIMULATION; the index and its thresholds were calibrated on Earth with the the earlier study body model; "
                      "on Mars and the Moon, and for this body model, they are uncalibrated",
               commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
               thresholds=dict(t1=T1, t2=T2), variants={k: list(v) for k, v in VARIANTS.items()},
               references=dict(table3=dict(refs=TABLE3_REFS, bands=TABLE3_BANDS),
                               ambient_test=json.load(open(os.path.join(ROOT, "outputs", "tables",
                                                                        "reference_modes.json")))["refs"]),
               display_grid=dict(fmin=0.3, fmax=50.0, n=750), earlier_baseline=EARLIER_BASELINE,
               scenarios=[{k: sc[k] for k in ("id", "label", "kind")} for sc in SCENARIOS], data={})
    for r in results:
        out["data"].setdefault(r["sid"], {})[r["planet"]] = r
    # class changes under the threshold variants (Table 3 references)
    cnt = {k: 0 for k in VARIANTS}
    tot = 0
    for r in results:
        for dev in ("watch", "phone"):
            cl = r["table3"][dev]["classes"]
            for m in cl["nominal"]:
                tot += 1
                for k in VARIANTS:
                    cnt[k] += cl[k][m] != cl["nominal"][m]
    out["threshold_sensitivity"] = dict(cells=tot, changed={k: v for k, v in cnt.items() if k != "nominal"})
    # references from this project's own ambient test instead of Table 3: class changes
    ch = sum(r["table3"][d]["pmode"][m]["band"] != r["ambient_test"][d]["pmode"].get(m, {}).get("band")
             for r in results for d in ("watch", "phone") for m in r["table3"][d]["pmode"])
    out["reference_sensitivity"] = dict(cells=tot, changed=int(ch))
    out["wall_seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "legacy_oi_baseline.json"), "w"), indent=1, default=float)
    # Markdown table
    labels = [m for m, _ in TABLE3_REFS]
    L = ["# Legacy Observability Index, the earlier study processing",
         "",
         "Mode-level index (0-100) and class (O observable >= 92.3, M marginal >= 27.5, U unobservable) for the "
         "watch (wrist) and phone (trouser pocket), from 20 records and their matched backgrounds, processed with "
         "the baseline's own code; references and bands of the earlier study, Table 3. The thresholds were calibrated on "
         "Earth for the earlier study's body model; for this body model and off Earth they are uncalibrated.",
         "",
         "| Situation | Planet | Device | " + " | ".join(labels) + " | Device |",
         "|---|---|---|" + "---|" * (len(labels) + 1)]
    lab = {sc["id"]: sc["label"] for sc in SCENARIOS}
    for r in results:
        for dev in ("watch", "phone"):
            d = r["table3"][dev]
            cells = [f"{d['pmode'][m]['oi']:.1f} {short(d['pmode'][m]['band'])}" for m in labels]
            L.append(f"| {lab[r['sid']]} | {r['planet']} | {dev} | " + " | ".join(cells)
                     + f" | {d['oi']:.1f} {short(d['band'])} |")
    L += ["", f"Threshold sensitivity: of {tot} mode cells, {cnt['lower25']} change class with both thresholds 25 % "
              f"lower and {cnt['higher25']} with both 25 % higher (upper capped at 100). With the references and bands "
              f"of this project's own ambient test instead of Table 3, {ch} cells change class."]
    open(os.path.join(ROOT, "outputs", "tables", "legacy_oi_baseline.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
