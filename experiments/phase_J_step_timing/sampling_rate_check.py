"""Phase J, post hoc (2026-11-21): accuracy of the device gait clock at the sampling
rates real consumer devices deliver.  The companion experiment (Luleci & Babazade,
the authors' building-floor study) recorded phones at about 62 Hz and
watches at about 90 Hz (resampled to 80 Hz); the simulated devices run at 200 Hz.

Each device signal (record and twin) is resampled to 62, 80 and 100 Hz (polyphase,
anti-aliased), the clock is estimated exactly as in step_timing.py at that rate, and
its error against the touchdown clock is computed on the 200 Hz time base.

Writes outputs/tables/step_timing_rates.json.

    python experiments/phase_J_step_timing/sampling_rate_check.py
"""
import glob
import json
import os
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from scipy.signal import resample_poly

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REC = os.path.join(ROOT, "outputs", "raw", "records")
RATES = [200.0, 100.0, 80.0, 62.0]
CONDS = ["earth_1.00", "mars_1.00", "moon_1.00", "earth3d_1.00", "earth_1.50"]
SENS = ["wrist_L", "thigh_L"]


def _resampled(X0, fs0, fs):
    fr = Fraction(fs / fs0).limit_denominator(100)
    if fs == fs0:
        return X0, fs0
    return resample_poly(X0, fr.numerator, fr.denominator, axis=0), fs0 * fr.numerator / fr.denominator


def one(job):
    """One condition, device position and rate, with the pooling of the declared method:
    events per stride and stride period pooled over the walks (records and twins)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import step_timing as ST
    from planetsense.simulation.records import crossing_window, load_record
    cond, sn, fs = job
    paths = sorted(glob.glob(os.path.join(REC, f"walking_{cond}_E1_r*.npz")))[:10] + \
        sorted(glob.glob(os.path.join(REC, f"walking_{cond}_rigid_r*.npz")))[:10]
    recs = [load_record(p) for p in paths]
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


def main():
    jobs = [(c, sn, fs) for c in CONDS for sn in SENS for fs in RATES]
    with Pool(4) as pool:
        res = pool.map(one, jobs)
    json.dump(dict(note="post hoc; shakers-on walking records (first 10 of each condition, with 10 rigid twins for "
                        "pooling)", rates=RATES, results=res),
              open(os.path.join(ROOT, "outputs", "tables", "step_timing_rates.json"), "w"), indent=1)
    for c in CONDS:
        for sn in SENS:
            rr = {x["rate"]: x for x in res if x["cond"] == c and x["sensor"] == sn}
            print(f"{c:14s} {sn:8s} random error (ms, median): " + "  ".join(
                f"{fs:g} Hz {rr[fs]['random_ms_median']:.2f} (n_e {rr[fs]['n_e']})" for fs in RATES))


if __name__ == "__main__":
    main()
