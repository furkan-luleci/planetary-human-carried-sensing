"""Phase J, post hoc (added 2026-11-20 after the first device-clock results; not part of
the declared criteria): how precise must the gait clock be for the earlier study's
order-tracked gait-harmonic removal?

The exact clock (touchdowns) is degraded in a controlled way: every touchdown time of a
record, and separately of its rigid-ground twin, is shifted by independent Gaussian
noise of standard deviation sigma (0.1, 0.3, 1 and 3 ms), and the clock is rebuilt from
the shifted touchdowns.  Everything else is the step-timing test's processing: index
(Table 3 references, nominal thresholds), band excess after removal, and blind recovery
after removal (M2) against chance.  sigma = 0 is the exact clock.

A diagnostic made on one case before this script (walking 1.0 m/s, footsteps only,
Earth) showed that the device clock's warp that repeats every stride does not matter:
adding it to the exact clock left the results unchanged, removing it from the device
clock did not restore them.  This script measures the effect of the part that does not
repeat.

Writes outputs/tables/step_timing_jitter.json.

    python experiments/phase_J_step_timing/jitter_sensitivity.py [--workers 4]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step_timing_test as STT  # noqa: E402

ROOT = STT.ROOT
SIGMAS_MS = [0.0, 0.1, 0.3, 1.0, 3.0]
CASES = [("walk_same_E1", "Earth"), ("walk_same_E2", "Earth"), ("baseline_walk", "Earth")]


def jittered_clock(rec, fp, sigma_s, seed):
    t = np.asarray(rec["t"], float)
    te = np.asarray(rec["events"], float).reshape(-1, 3)[:, 0].copy()
    if sigma_s > 0:
        te = te + np.random.default_rng(seed).normal(0.0, sigma_s, len(te))
        te = np.maximum.accumulate(te)
    j = np.arange(len(te), dtype=float)
    psi = np.interp(t, te, j)
    lo, hi = t < te[0], t > te[-1]
    psi[lo] = (t[lo] - te[0]) / (te[1] - te[0])
    psi[hi] = j[-1] + (t[hi] - te[-1]) / (te[-1] - te[-2])
    return psi / float(fp)


def run(arg):
    sid, planet, sigma_ms = arg
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import zlib
    from planetsense.evaluation import legacy_pipeline as LP
    from planetsense.evaluation import baseline_match as MM
    from planetsense.simulation.records import load_record
    t0 = time.time()
    tk = [t for t in STT.walking_tasks() if t["sid"] == sid and t["planet"] == planet][0]
    recs = [load_record(f) for f in tk["files"]]
    twins = [load_record(f) for f in tk["twins"]]
    mv = json.load(open(os.path.join(STT.MI.GAIT, tk["cond"] + ".json")))["metrics_validation_30s"]
    fp, speed = float(mv["step_frequency"]), float(mv["speed"])
    _, Tres = MM.record_window_s(False, speed)
    O = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json")))
    rule, truth = O["recovery_rule"]["settings"], [m["f"] for m in O["modes"]]
    fstr_dev = O["data"][sid][planet]["device_stride"]
    bands = dict(zip([m for m, _ in STT.MI.TABLE3_REFS], STT.MI.TABLE3_BANDS))
    g = np.linspace(0.3, 50.0, 750)
    sig = 1e-3 * sigma_ms
    seed = lambda r, side: zlib.crc32(f"{r['name']}|{side}|jitter|{sigma_ms}".encode())
    sides = {}
    for dev, pairs in STT.SENSORS.items():
        sides[dev] = []
        for ours, base in pairs:
            cr = [jittered_clock(r, fp, sig, seed(r, ours)) for r in recs]
            cb = [jittered_clock(r, fp, sig, seed(r, ours)) for r in twins]
            E = MM.ensemble(recs, False, fp=fp, sensors=[base], clocks=cr)
            B = MM.ensemble(twins, False, fp=fp, sensors=[base], clocks=cb)
            sides[dev].append((E, B))
    with STT._Capture(LP) as cap:
        curves = MM.device_curves_sided(sides)
    r = MM.index_run(curves, STT.MI.TABLE3_REFS, STT.MI.TABLE3_BANDS, Tres)
    out = dict(case=sid, planet=planet, sigma_ms=sigma_ms, devices={})
    for dev, pairs in STT.SENSORS.items():
        _, S_, B_ = MM.display(curves[dev])
        cf = [[cap.store[(id(E), i, base)] for i in range(len(E["passes"]))] for (E, B), (_o, base) in zip(sides[dev], pairs)]
        cb = [[cap.store[(id(B), i, base)] for i in range(len(B["passes"]))] for (E, B), (_o, base) in zip(sides[dev], pairs)]
        f, S = STT.m2_spectra(cf, MM.FS_IMU)
        _, Bs = STT.m2_spectra(cb, MM.FS_IMU)
        key = "wrist" if dev == "watch" else "thigh"
        m2 = STT.m2_recover(f, S, Bs, rule, truth, fstr_dev[key]["f"])
        out["devices"][dev] = dict(
            classes={m: STT.SHORT[v["band"]] for m, v in r[dev]["pmode"].items()},
            oi={m: float(v["oi"]) for m, v in r[dev]["pmode"].items()},
            excess_db=STT.band_excess(S_, B_, bands, g),
            m2={k: dict(tp=v["tp"], fp=v["fp"], chance=v["chance_tp"], found=v["found"]) for k, v in m2.items()})
    out["wall"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(sid, p, s) for sid, p in CASES for s in SIGMAS_MS]
    with Pool(a.workers) as pool:
        res = list(pool.imap(run, jobs))
    C = json.load(open(os.path.join(ROOT, "outputs", "tables", "step_timing_clocks.json")))
    dev_random = {}
    for row in C["rows"]:
        if (row["sid"], row["planet"]) in CASES:
            dev_random.setdefault(f"{row['sid']}|{row['planet']}", {})[row["sensor"]] = dict(
                random_ms_median=row["random_ms_median"], total_ms_median=row["total_ms_median"])
    json.dump(dict(note="post hoc; not part of the declared criteria (docs/phase_reports/PHASE_J_step_timing.md)",
                   sigmas_ms=SIGMAS_MS, cases=[list(c) for c in CASES], results=res,
                   device_clock_error_ms=dev_random, wall_seconds=time.time() - t0),
              open(os.path.join(ROOT, "outputs", "tables", "step_timing_jitter.json"), "w"), indent=1, default=float)
    for r in res:
        line = f"{r['case']:13s} {r['planet']:5s} sigma {r['sigma_ms']:4.1f} ms |"
        for dev, d in r["devices"].items():
            line += (f" {dev}: " + "".join(d["classes"].values()) + " excess B1/B2/T1 "
                     + "/".join(f"{d['excess_db'][m]:.1f}" for m in ("B1", "B2", "T1"))
                     + f" M2 {d['m2']['no_rejection']['tp']} vs {d['m2']['no_rejection']['chance']} |")
        print(line, flush=True)
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
