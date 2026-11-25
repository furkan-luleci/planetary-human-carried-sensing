"""Phase J: the earlier study's index with gait timing estimated by each worn
device itself instead of the exact gait clock.

The baseline processing (vendored in src/planetsense/evaluation/legacy_pipeline.py)
removes the gait's harmonics by order tracking on the exact gait phase.  Here
every device (left and right wrist, left and right trouser pocket) estimates its
own gait clock from its own signal (src/planetsense/evaluation/step_timing.py), on the
record and, separately, on its rigid-ground twin; everything else is unchanged:
same records, same matched sensor noise in the processing, same harmonic
detection and removal, cross-spectra, Eq. 7 combination of the two sides,
smoothing, display grid, index, references and thresholds (the earlier study, Table 3,
27.5 / 92.3).

Each side is processed as a single-side ensemble so that it can carry its own
clock; the first singular values of the two sides are then averaged exactly as
the baseline does within one ensemble.  The same per-side path with the exact
clock is run alongside, and must reproduce outputs/tables/legacy_oi_baseline.json.

Estimator inputs: the device's specific force over the whole record plus white
noise of the same density as the processing (75 ug/sqrt(Hz)), seeded by record
and sensor.  The number of events per stride and the nominal step frequency
(2 / stride period) are pooled over the records and twins of each walking case
and device position, so a record and its twin share the nominal step frequency
(the baseline's pairing check).

Blind recovery after order-tracked removal (M2): the processed channels of the
vendored code (noise, trim, removal with the common harmonic set of record and
twin) -> cross-spectral matrix per side over the records with 4 s segments (as
the main analysis) -> first singular value, mean of the sides -> the recovery
rule of the main analysis (outputs/tables/outcomes.json, as frozen after the
Phase J correction) with its harmonic rejection off (primary: the removal takes
its place) and on (stride frequency from the device clock) -> one-to-one match
to the bridge's natural frequencies below 30 Hz; chance: the same on the
rigid-ground twins.  Variants: no removal, exact clock, device clock.

Writes outputs/tables/step_timing.json.  The criteria declared in
docs/phase_reports/PHASE_J_step_timing.md are evaluated by
experiments/phase_J_step_timing/check_step_timing.py.

    python experiments/phase_J_step_timing/step_timing_test.py [--workers 4] [--check]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_I_baseline_match"))
import baseline_index as MI  # noqa: E402  (task list, references and thresholds of Phase I)

SENSORS = {"watch": [("wrist_L", "wristL"), ("wrist_R", "wristR")],
           "phone": [("thigh_L", "hipL"), ("thigh_R", "hipR")]}
NOISE_UG = 75.0
SHORT = {"observable": "O", "marginally observable": "M", "unobservable": "U"}


def estimator_input(r, sensor):
    from planetsense.evaluation.step_timing import device_input
    return device_input(r, sensor, NOISE_UG)


def device_clocks(recs, twins):
    """Per sensor: events per stride, stride period, nominal step frequency, and the
    device clock of every record and twin; accuracy against the touchdown clock."""
    from planetsense.evaluation import step_timing as ST
    from planetsense.simulation.records import crossing_window
    allr = list(recs) + list(twins)
    out = {}
    for dev, pairs in SENSORS.items():
        for ours, _base in pairs:
            Xs, ev, sc = [], [], []
            for r in allr:
                X = estimator_input(r, ours)
                te, Tev = ST.detected_events(X, float(r["fs"]))
                Xs.append(X)
                ev.append(Tev)
                sc.append(ST.stride_scores(X, float(r["fs"]), Tev))
            n_e, tot = ST.choose_events_per_stride(sc)
            T_str = float(np.median([n_e * v for v in ev]))
            fp = 2.0 / T_str
            clocks, acc = [], []
            for r, X in zip(allr, Xs):
                fs = float(r["fs"])
                psi_e, te_used, _k = ST.device_event_count(X, fs, n_e, 1.0 / T_str)
                clocks.append(ST.device_gait_clock(psi_e, n_e, fp))
                m = crossing_window(r)
                T_true = 2.0 * float(np.median(np.diff(np.asarray(r["events"]).reshape(-1, 3)[:, 0])))
                tot_ms, rnd_ms, max_ms = ST.timing_error(psi_e[m] / n_e, ST.true_stride_phase(r)[m], T_true)
                acc.append(dict(name=r["name"], total_ms=tot_ms, random_ms=rnd_ms, max_ms=max_ms,
                                events_used=int(len(te_used)), true_stride_s=T_true))
            out[ours] = dict(n_e=int(n_e), scores={str(k): v for k, v in tot.items()}, stride_s=T_str, fp=fp,
                             clocks_rec=clocks[:len(recs)], clocks_twin=clocks[len(recs):], accuracy=acc)
    return out


class _Capture:
    """Keeps the processed channels the vendored code computes (device_channels), keyed by
    (ensemble, record index, sensor), so that the same signals feed the M2 spectra."""

    def __init__(self, LP):
        self.LP, self.orig, self.store = LP, LP.device_channels, {}

    def __enter__(self):
        def wrap(ens, p, i, sensor, noise_ug, comb_lines=None, comb_out=None):
            out = self.orig(ens, p, i, sensor, noise_ug, comb_lines=comb_lines, comb_out=comb_out)
            self.store[(id(ens), int(i), str(sensor))] = np.array(out, float)
            return out
        self.LP.device_channels = wrap
        return self

    def __exit__(self, *a):
        self.LP.device_channels = self.orig


def raw_channels(LP, ens, sensor):
    """The vendored code's channels before harmonic removal (mean and trend removed, sensor
    noise added, first 1.5 s trimmed)."""
    nz = LP.sensor_noise_of(ens["cfg"], sensor)
    out = []
    for i, p in enumerate(ens["passes"]):
        out.append(np.array([LP._device_axis_signal(ens, p, i, sensor, nz, a, ax)[0]
                             for a, ax in enumerate(LP.AXIS_NAMES)]))
    return out


def m2_spectra(chans_by_side, fs, nper_s=4.0):
    """First-singular-value spectrum of a device: per side, the cross-spectral matrix over
    the records (4 s segments), its first singular value; mean of the sides."""
    from planetsense.identification.spectral import ensemble_csd, sv1
    acc, f = None, None
    for chans in chans_by_side:
        f, G, _ = ensemble_csd([c.T for c in chans], fs, int(nper_s * fs))
        acc = sv1(G) if acc is None else acc + sv1(G)
    return f, acc / len(chans_by_side)


def m2_recover(f, S, Bs, rule, truth, f_stride):
    from planetsense.identification.spectral import match_peaks, pick_peaks
    out = {}
    for name, fh in (("no_rejection", None), ("rejection", f_stride)):
        res = []
        for spec in (S, Bs):
            pk = pick_peaks(f, spec, fmin=1.0, fmax=30.0, prominence_db=rule["prom"], background_hz=rule["bg"],
                            harmonics_of=fh, harmonic_halfwidth=rule["hw"], max_harmonic_hz=rule["hmax"])
            res.append(match_peaks([q[0] for q in pk], truth, tol_rel=0.03, tol_abs=0.15))
        out[name] = dict(tp=int(res[0]["tp"]), fp=int(res[0]["fp"]), found=[float(q[0]) for q in res[0]["pairs"]],
                         chance_tp=int(res[1]["tp"]), chance_fp=int(res[1]["fp"]),
                         chance_found=[float(q[0]) for q in res[1]["pairs"]])
    return out


def band_excess(S, B, bands, g):
    out = {}
    for m, (a, b) in bands.items():
        k = (g >= a) & (g <= b)
        if k.sum() < 2:
            k = np.abs(g - 0.5 * (a + b)) <= max(0.5 * (b - a), g[1] - g[0])
        out[m] = float(10 * np.log10(np.nanmean(S[k]) / np.nanmean(B[k])))
    return out


def run_task(tk, check_only=False):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import baseline_match as MM
    from planetsense.simulation.records import load_record
    t0 = time.time()
    recs = [load_record(f) for f in tk["files"]]
    twins = [load_record(f) for f in tk["twins"]]
    keep = [i for i in range(len(recs)) if not recs[i].get("fell") and not twins[i].get("fell")]
    recs, twins = [recs[i] for i in keep], [twins[i] for i in keep]
    mv = json.load(open(os.path.join(MI.GAIT, tk["cond"] + ".json")))["metrics_validation_30s"]
    fp_nom, speed = float(mv["step_frequency"]), float(mv["speed"])
    Trec, Tres = MM.record_window_s(False, speed)
    bands = dict(zip([m for m, _ in MI.TABLE3_REFS], MI.TABLE3_BANDS))
    g = np.linspace(0.3, 50.0, 750)
    clocks = None if check_only else device_clocks(recs, twins)
    from planetsense.evaluation import legacy_pipeline as LP
    O = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json")))
    rule, truth = O["recovery_rule"]["settings"], [m["f"] for m in O["modes"]]
    res = dict(sid=tk["sid"], planet=tk["planet"], cond=tk["cond"], n_records=len(recs), fp_nominal=fp_nom,
               speed=speed, Tres=Tres, m2=dict(rule=rule, truth=truth))
    for variant in (("exact",) if check_only else ("exact", "device")):
        sides = {}
        for dev, pairs in SENSORS.items():
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
        with _Capture(LP) as cap:
            curves = MM.device_curves_sided(sides)
        r = MM.index_run(curves, MI.TABLE3_REFS, MI.TABLE3_BANDS, Tres)
        if not check_only:
            m2v = {"none": {}, variant: {}} if variant == "exact" else {variant: {}}
            for dev, pairs in SENSORS.items():
                fstr = float(np.mean([1.0 / clocks[o]["stride_s"] for o, _b in pairs]))
                for vname in m2v:
                    cf, cb = [], []
                    for (E, B), (_o, base) in zip(sides[dev], pairs):
                        if vname == "none":
                            cf.append(raw_channels(LP, E, base))
                            cb.append(raw_channels(LP, B, base))
                        else:
                            cf.append([cap.store[(id(E), i, base)] for i in range(len(E["passes"]))])
                            cb.append([cap.store[(id(B), i, base)] for i in range(len(B["passes"]))])
                    f, S = m2_spectra(cf, MM.FS_IMU)
                    _, Bs = m2_spectra(cb, MM.FS_IMU)
                    k = f <= 30.0
                    m2v[vname][dev] = dict(m2_recover(f, S, Bs, rule, truth, fstr), f_stride=fstr,
                                           f=[round(float(v), 4) for v in f[k]],
                                           S=[float(f"{v:.4g}") for v in S[k]], B=[float(f"{v:.4g}") for v in Bs[k]])
            res["m2"].update(m2v)
        res[variant] = {}
        for dev in SENSORS:
            d = r[dev]
            gg, S, Bc = MM.display(curves[dev])
            res[variant][dev] = dict(
                oi=float(d["oi"]), band=d["band"], n_comb_lines=curves[dev]["n_comb_lines"],
                pmode={m: dict(oi=float(v["oi"]), cls=SHORT[v["band"]], f=v["f"], F=v["F"], Po=v["Po"], Pb=v["Pb"])
                       for m, v in d["pmode"].items()},
                excess_db=band_excess(S, Bc, bands, g),
                S=[float(f"{v:.4g}") for v in S], B=[float(f"{v:.4g}") for v in Bc])
    if clocks is not None:
        res["timing"] = {s: {k: v for k, v in c.items() if k not in ("clocks_rec", "clocks_twin")}
                         for s, c in clocks.items()}
    res["wall"] = time.time() - t0
    return res


def _check(res):
    """Per-side exact path against the Phase I result (the same numbers expected)."""
    L = json.load(open(os.path.join(ROOT, "outputs", "tables", "legacy_oi_baseline.json")))
    ref = L["data"][res["sid"]][res["planet"]]["table3"]
    dmax, mism = 0.0, 0
    for dev in SENSORS:
        for m, v in res["exact"][dev]["pmode"].items():
            dmax = max(dmax, abs(v["oi"] - ref[dev]["pmode"][m]["oi"]))
            mism += v["cls"] != SHORT[ref[dev]["pmode"][m]["band"]]
        dmax = max(dmax, abs(res["exact"][dev]["oi"] - ref[dev]["oi"]))
    return dict(max_abs_oi_difference=dmax, class_mismatches=int(mism))


def walking_tasks():
    return [tk for tk in MI.tasks() if tk["kind"] == "walking"]


def _run(tk):
    return run_task(tk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true",
                    help="only the exact clock on one case, compared with Phase I (no estimated timing)")
    a = ap.parse_args()
    t0 = time.time()
    tks = walking_tasks()
    if a.check:
        tk = [t for t in tks if t["sid"] == "walk_same_E1" and t["planet"] == "Earth"][0]
        r = run_task(tk, check_only=True)
        print("per-side exact path vs Phase I:", _check(r), f"({r['wall']:.0f} s)")
        return
    print(len(tks), "walking cases", flush=True)
    with Pool(a.workers) as pool:
        results = list(pool.imap(_run, tks))
    out = dict(status="SIMULATION; index and thresholds of the earlier study (calibrated on Earth for its body model); "
                      "uncalibrated for this body model and off Earth",
               commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
               declared="docs/phase_reports/PHASE_J_step_timing.md",
               thresholds=dict(t1=MI.T1, t2=MI.T2), references=dict(refs=MI.TABLE3_REFS, bands=MI.TABLE3_BANDS),
               display_grid=dict(fmin=0.3, fmax=50.0, n=750), noise_ug=NOISE_UG, data={})
    for r in results:
        r["regression_vs_phase_I"] = _check(r)
        out["data"].setdefault(r["sid"], {})[r["planet"]] = r
    out["wall_seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "step_timing.json"), "w"), indent=1, default=float)
    for r in results:
        line = f"{r['sid']:16s} {r['planet']:5s} reg {r['regression_vs_phase_I']} |"
        for dev in SENSORS:
            line += f" {dev}: " + "".join(r["exact"][dev]["pmode"][m]["cls"] for m in r["exact"][dev]["pmode"]) + " -> " + \
                    "".join(r["device"][dev]["pmode"][m]["cls"] for m in r["device"][dev]["pmode"]) + " |"
        print(line, flush=True)
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
