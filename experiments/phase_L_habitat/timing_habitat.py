"""Phase L (2026-11-21): the device gait clock on the habitat floor, with the same methods as on
the reference deck (experiments/phase_J_step_timing/): accuracy of every device clock, how fast
the gait-harmonic removal degrades when random errors are added to the exact touchdown times,
and the clock's random error at consumer sampling rates.  Post hoc: the habitat floor became the
study's main case after these analyses had been run on the reference deck only.  No new
simulation; the 300 habitat-floor records and the reference deck's rigid-ground twins (cut to
the floor's crossing) are reused, as in experiments/phase_K_S2/step_timing_s2.py.

Writes outputs/tables/step_timing_clocks_<S>.json, step_timing_jitter_<S>.json and
step_timing_rates_<S>.json.

    python experiments/phase_L_habitat/timing_habitat.py [S2] [--workers 4] [--parts clocks,jitter,rates]
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from scipy.signal import resample_poly

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_J_step_timing"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_I_baseline_match"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_K_S2"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step_timing_test as STT  # noqa: E402
import step_timing_s2 as S2T  # noqa: E402
from jitter_sensitivity import SIGMAS_MS, jittered_clock  # noqa: E402

TAB = os.path.join(ROOT, "outputs", "tables")
JITTER_CASES = [("walk_same_E1", "Earth"), ("walk_same_E2", "Earth"), ("walk_same_E1", "Moon"), ("walk_same_E2", "Moon")]
RATES = [200.0, 100.0, 80.0, 62.0]
RATE_CONDS = ["earth_1.00", "mars_1.00", "moon_1.00", "earth3d_1.00"]
RATE_SENS = ["wrist_L", "thigh_L"]


def _structure(struct):
    from planetsense.structure import registry as SR
    md = SR.modal_deck(struct)
    modes = [dict(label=l, f=float(q)) for l, q in zip(md.labels, md.freq_hz) if q < 30.0]
    return float(md.params.L), modes


def _occupied(struct, planet):
    occ = json.load(open(os.path.join(TAB, "occupied_modes.json")))["data"][struct]
    return {l: (v["f"] if v else occ["empty"][l]) for l, v in occ["occupied"][planet].items()}


def _load_pairs(tk, L):
    from planetsense.simulation.records import load_record
    recs = [dict(load_record(f), deck_L=L) for f in tk["files"]]
    twins = [dict(load_record(f), deck_L=L) for f in tk["twins"]]
    keep = [i for i in range(len(recs)) if not recs[i].get("fell") and not twins[i].get("fell")]
    return [recs[i] for i in keep], [twins[i] for i in keep]


# ------------------------------------------------------------------ clocks
def clocks_job(tk):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    L, _ = _structure(tk["struct"])
    recs, twins = _load_pairs(tk, L)
    c = STT.device_clocks(recs, twins)
    return dict(sid=tk["sid"], planet=tk["planet"], cond=tk["cond"],
                sensors={s: {k: v for k, v in d.items() if k not in ("clocks_rec", "clocks_twin")} for s, d in c.items()})


def clocks(struct, workers):
    t0 = time.time()
    with Pool(workers) as pool:
        res = list(pool.imap(clocks_job, S2T.tasks(struct)))
    rows = []
    for r in res:
        for s, d in r["sensors"].items():
            A = d["accuracy"]
            tot = np.array([x["total_ms"] for x in A]); rnd = np.array([x["random_ms"] for x in A])
            T = np.median([x["true_stride_s"] for x in A])
            rows.append(dict(sid=r["sid"], planet=r["planet"], cond=r["cond"], sensor=s, n_e=d["n_e"],
                             stride_s=d["stride_s"], stride_true_s=float(T),
                             stride_error_pct=float(100 * (d["stride_s"] - T) / T),
                             total_ms_median=float(np.median(tot)), total_ms_p95=float(np.percentile(tot, 95)),
                             random_ms_median=float(np.median(rnd)), random_ms_p95=float(np.percentile(rnd, 95)),
                             random_ms_max=float(rnd.max()), records_over_30ms=int((tot > 30).sum()), records=len(A)))
    allr = [x for r in res for d in r["sensors"].values() for x in d["accuracy"]]
    tot = np.array([x["total_ms"] for x in allr]); rnd = np.array([x["random_ms"] for x in allr])
    summ = dict(clock_estimates=len(allr), total_ms_median=float(np.median(tot)), total_ms_p95=float(np.percentile(tot, 95)),
                random_ms_median=float(np.median(rnd)), random_ms_p95=float(np.percentile(rnd, 95)),
                over_30ms=int((tot > 30).sum()),
                wrong_events_per_stride=int(sum(abs(x["stride_error_pct"]) > 10 for x in rows)),
                case_positions=len(rows))
    json.dump(dict(structure=struct, rows=rows, summary=summ, cases=res, wall_seconds=time.time() - t0),
              open(os.path.join(TAB, f"step_timing_clocks_{struct}.json"), "w"), indent=1, default=float)
    print("clocks", struct, summ, flush=True)


# ------------------------------------------------------------------ jitter
def jitter_job(arg):
    struct, sid, planet, sigma_ms = arg
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import legacy_pipeline as LP
    from planetsense.evaluation import baseline_match as MM
    t0 = time.time()
    L, modes = _structure(struct)
    occ_p = _occupied(struct, planet)
    tk = [t for t in S2T.tasks(struct) if t["sid"] == sid and t["planet"] == planet][0]
    recs, twins = _load_pairs(tk, L)
    mv = json.load(open(os.path.join(ROOT, "outputs", "gait", tk["cond"] + ".json")))["metrics_validation_30s"]
    fp = float(mv["step_frequency"])
    rule = json.load(open(os.path.join(TAB, "outcomes.json")))["recovery_rule"]["settings"]
    fstr_dev = json.load(open(os.path.join(TAB, f"outcomes_{struct}.json")))["data"][sid][planet]["device_stride"]
    # bands: each mode's band (+-max(0.15 Hz, 3 %)) widened to cover its empty and occupied frequencies
    dl = lambda f: max(0.15, 0.03 * f)
    bands = {m["label"]: (min(m["f"], occ_p[m["label"]]) - dl(m["f"]), max(m["f"], occ_p[m["label"]]) + dl(occ_p[m["label"]]))
             for m in modes}
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
    out = dict(case=sid, planet=planet, sigma_ms=sigma_ms, bands=bands, occupied=occ_p, devices={})
    for dev, pairs in STT.SENSORS.items():
        _, S_, B_ = MM.display(curves[dev])
        cf = [[cap.store[(id(E), i, base)] for i in range(len(E["passes"]))] for (E, B), (_o, base) in zip(sides[dev], pairs)]
        cb = [[cap.store[(id(B), i, base)] for i in range(len(B["passes"]))] for (E, B), (_o, base) in zip(sides[dev], pairs)]
        f, S = STT.m2_spectra(cf, MM.FS_IMU)
        _, Bs = STT.m2_spectra(cb, MM.FS_IMU)
        key = "wrist" if dev == "watch" else "thigh"
        m2 = S2T.m2_recover2(f, S, Bs, rule, modes, occ_p, fstr_dev[key]["f"])
        out["devices"][dev] = dict(excess_db=STT.band_excess(S_, B_, bands, g),
                                   m2={k: dict(tp=v["tp"], fp=v["fp"], chance=v["chance_tp"], found=v["matched"])
                                       for k, v in m2.items()})
    out["wall"] = time.time() - t0
    return out


def jitter(struct, workers):
    t0 = time.time()
    jobs = [(struct, sid, p, s) for sid, p in JITTER_CASES for s in SIGMAS_MS]
    with Pool(workers) as pool:
        res = list(pool.imap(jitter_job, jobs))
    C = json.load(open(os.path.join(TAB, f"step_timing_clocks_{struct}.json")))
    dev_random = {}
    for row in C["rows"]:
        if (row["sid"], row["planet"]) in JITTER_CASES:
            dev_random.setdefault(f"{row['sid']}|{row['planet']}", {})[row["sensor"]] = dict(
                random_ms_median=row["random_ms_median"], total_ms_median=row["total_ms_median"])
    json.dump(dict(structure=struct, note="post hoc (Phase L); method of experiments/phase_J_step_timing/jitter_sensitivity.py; "
                   "bands cover each mode's empty and declared occupied frequency",
                   sigmas_ms=SIGMAS_MS, cases=[list(c) for c in JITTER_CASES], results=res,
                   device_clock_error_ms=dev_random, wall_seconds=time.time() - t0),
              open(os.path.join(TAB, f"step_timing_jitter_{struct}.json"), "w"), indent=1, default=float)
    for r in res:
        print(f"jitter {r['case']:13s} {r['planet']:5s} sigma {r['sigma_ms']:4.1f} ms | " + " | ".join(
            f"{dev}: excess " + "/".join(f"{v:.1f}" for v in d["excess_db"].values())
            + f" M2 {d['m2']['no_rejection']['tp']} vs {d['m2']['no_rejection']['chance']}" for dev, d in r["devices"].items()),
            flush=True)


# ------------------------------------------------------------------ sampling rates
def _resampled(X0, fs0, fs):
    if fs == fs0:
        return X0, fs0
    fr = Fraction(fs / fs0).limit_denominator(100)
    return resample_poly(X0, fr.numerator, fr.denominator, axis=0), fs0 * fr.numerator / fr.denominator


def rate_job(job):
    """As experiments/phase_J_step_timing/sampling_rate_check.py: first 10 shakers-on walks and 10 rigid twins."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import step_timing as ST
    from planetsense.simulation.records import crossing_window, load_record
    struct, cond, sn, fs = job
    L, _ = _structure(struct)
    paths = sorted(glob.glob(os.path.join(ROOT, "outputs", "raw", "records_" + struct, f"walking_{cond}_E1_r*.npz")))[:10] + \
        sorted(glob.glob(os.path.join(ROOT, "outputs", "raw", "records", f"walking_{cond}_rigid_r*.npz")))[:10]
    recs = [dict(load_record(p), deck_L=L) for p in paths]
    Xs, Tevs, scs = [], [], []
    for r in recs:
        X, fsx = _resampled(ST.device_input(r, sn), float(r["fs"]), fs)
        _, Tev = ST.detected_events(X, fsx)
        Xs.append((X, fsx)); Tevs.append(Tev); scs.append(ST.stride_scores(X, fsx, Tev))
    n_e, _ = ST.choose_events_per_stride(scs)
    T_str = float(np.median([n_e * v for v in Tevs]))
    rnd, tot = [], []
    for r, (X, fsx) in zip(recs, Xs):
        if "_E1_" not in r["name"]:
            continue
        psi, _, _ = ST.device_event_count(X, fsx, n_e, 1.0 / T_str)
        t = np.asarray(r["t"], float); m = crossing_window(r)
        ph = np.interp(t - t[0], np.arange(len(X)) / fsx, psi) / n_e
        T = 2.0 * float(np.median(np.diff(np.asarray(r["events"]).reshape(-1, 3)[:, 0])))
        a, b, _ = ST.timing_error(ph[m], ST.true_stride_phase(r)[m], T)
        tot.append(a); rnd.append(b)
    return dict(cond=cond, sensor=sn, rate=fs, n_e=int(n_e), stride_s=T_str,
                random_ms_median=float(np.median(rnd)), total_ms_median=float(np.median(tot)), n=len(rnd))


def rates(struct, workers):
    jobs = [(struct, c, sn, fs) for c in RATE_CONDS for sn in RATE_SENS for fs in RATES]
    with Pool(workers) as pool:
        res = pool.map(rate_job, jobs)
    json.dump(dict(structure=struct, note="post hoc (Phase L); shakers-on walking records (first 10 of each condition, with 10 "
                   "rigid twins for pooling), method of experiments/phase_J_step_timing/sampling_rate_check.py",
                   rates=RATES, results=res),
              open(os.path.join(TAB, f"step_timing_rates_{struct}.json"), "w"), indent=1)
    for c in RATE_CONDS:
        for sn in RATE_SENS:
            rr = {x["rate"]: x for x in res if x["cond"] == c and x["sensor"] == sn}
            print(f"rates {c:14s} {sn:8s} random error (ms, median): " + "  ".join(
                f"{fs:g} Hz {rr[fs]['random_ms_median']:.2f} (n_e {rr[fs]['n_e']})" for fs in RATES), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("struct", nargs="?", default="S2", choices=("S2", "S2L"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parts", default="clocks,jitter,rates")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if "clocks" in parts:
        clocks(a.struct, a.workers)
    if "jitter" in parts:
        jitter(a.struct, a.workers)
    if "rates" in parts:
        rates(a.struct, a.workers)


if __name__ == "__main__":
    main()
