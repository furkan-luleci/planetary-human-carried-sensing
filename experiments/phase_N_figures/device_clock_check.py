"""Post hoc check (2026-11-22, added after the review of the signal figures; not part of any declared
criterion): why does removal on the device's own gait clock lose so little at B2 in fig_signals_freq (e),
when 0.3 ms of added random touchdown error already costs 3.8 dB at B2 in Fig. steptiming b?

One walk only: habitat floor, side-view walker, Earth, shakers on (walk_same_E1), the walk of
fig_signals_freq.  The heights of B1 and B2 above rigid ground after the earlier study's gait-harmonic removal
are computed on both measures in use, in the bands of experiments/phase_L_habitat/timing_habitat.py (each
mode's band widened to cover its empty and its declared occupied frequency):

  display  the earlier study's device curves (step_timing_jitter_S2.json, Fig. steptiming b)
  m2       the first-singular-value spectra on which blind recovery runs (step_timing_S2.json,
           fig_signals_freq e and h)

for six clock variants:

  jit0.0    exact clock on the records and on their rigid-ground twins
  jit0.3    the exact touchdowns with 0.3 ms of independent Gaussian error, records and twins (as Fig.
            steptiming b)
  jit1.0    the same with 1 ms
  device    each device's own clock on the records and on the twins (as step_timing_S2.json)
  hyb_rec   the device's own clock on the records, the exact clock on the twins
  hyb_twin  the exact clock on the records, the device's own clock on the twins

For every variant and device the result keeps both heights, the band powers of the record (S) and of the
twin (B) on the m2 spectra, and the median of the dB difference outside the bands (3-30 Hz, at least 0.5 Hz
from either band).  jit0.0, jit0.3, jit1.0 and device reproduce the stored results (step_timing_jitter_S2.json
and step_timing_S2.json); the script asserts it.

Writes outputs/tables/device_clock_check_S2.json (about 15 min with 4 workers).

    PYTHONPATH=src python experiments/phase_N_figures/device_clock_check.py [--workers 4]
"""
import argparse
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for d in ("phase_J_step_timing", "phase_I_baseline_match", "phase_K_S2", "phase_L_habitat"):
    sys.path.insert(0, os.path.join(ROOT, "experiments", d))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step_timing_test as STT  # noqa: E402
import step_timing_s2 as S2T  # noqa: E402
import timing_habitat as TH  # noqa: E402
from jitter_sensitivity import jittered_clock  # noqa: E402

TAB = os.path.join(ROOT, "outputs", "tables")
STRUCT, SID, PLANET = "S2", "walk_same_E1", "Earth"
VARIANTS = ["jit0.0", "jit0.3", "jit1.0", "device", "hyb_rec", "hyb_twin"]


def _bands(modes, occ_p):
    dl = lambda f: max(0.15, 0.03 * f)
    return {m["label"]: (min(m["f"], occ_p[m["label"]]) - dl(m["f"]), max(m["f"], occ_p[m["label"]]) + dl(occ_p[m["label"]]))
            for m in modes}


def _off_band(f, S, B, bands):
    k = (f >= 3.0) & (f <= 30.0)
    for a, b in bands.values():
        k &= ~((f >= a - 0.5) & (f <= b + 0.5))
    return float(np.median(10.0 * np.log10(S[k] / B[k])))


def job(variant):
    try:
        return _job(variant)
    except Exception as ex:                                   # keep the other variants
        return dict(variant=variant, error=repr(ex))


def _job(variant):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.evaluation import legacy_pipeline as LP
    from planetsense.evaluation import baseline_match as MM
    t0 = time.time()
    L, modes = TH._structure(STRUCT)
    occ_p = TH._occupied(STRUCT, PLANET)
    tk = [t for t in S2T.tasks(STRUCT) if t["sid"] == SID and t["planet"] == PLANET][0]
    recs, twins = TH._load_pairs(tk, L)
    mv = json.load(open(os.path.join(ROOT, "outputs", "gait", tk["cond"] + ".json")))["metrics_validation_30s"]
    fp = float(mv["step_frequency"])
    bands = _bands(modes, occ_p)
    g = np.linspace(0.3, 50.0, 750)
    clocks = STT.device_clocks(recs, twins) if variant in ("device", "hyb_rec", "hyb_twin") else None

    def exact_or_jittered(rs, sig_ms, ours, f_step):
        """As experiments/phase_J_step_timing/jitter_sensitivity.py (same seeds); with sig_ms = 0 the exact clock
        in units of the step rate ``f_step``."""
        seed = lambda r: zlib.crc32(f"{r['name']}|{ours}|jitter|{sig_ms}".encode())
        return [jittered_clock(r, f_step, 1e-3 * sig_ms, seed(r)) for r in rs]

    sides = {}
    for dev, pairs in STT.SENSORS.items():
        sides[dev] = []
        for ours, base in pairs:
            if variant.startswith("jit"):
                s = float(variant[3:])
                E = MM.ensemble(recs, False, fp=fp, sensors=[base], clocks=exact_or_jittered(recs, s, ours, fp))
                B = MM.ensemble(twins, False, fp=fp, sensors=[base], clocks=exact_or_jittered(twins, s, ours, fp))
            else:                                             # a record and its twin need the same step rate
                c = clocks[ours]
                cr = c["clocks_rec"] if variant in ("device", "hyb_rec") else exact_or_jittered(recs, 0.0, ours, c["fp"])
                cb = c["clocks_twin"] if variant in ("device", "hyb_twin") else exact_or_jittered(twins, 0.0, ours, c["fp"])
                E = MM.ensemble(recs, False, fp=c["fp"], sensors=[base], clocks=cr)
                B = MM.ensemble(twins, False, fp=c["fp"], sensors=[base], clocks=cb)
            sides[dev].append((E, B))
    with STT._Capture(LP) as cap:
        curves = MM.device_curves_sided(sides)
    out = dict(variant=variant, bands=bands, devices={})
    for dev, pairs in STT.SENSORS.items():
        _, S_, B_ = MM.display(curves[dev])
        cf = [[cap.store[(id(E), i, base)] for i in range(len(E["passes"]))] for (E, B), (_o, base) in zip(sides[dev], pairs)]
        cb = [[cap.store[(id(B), i, base)] for i in range(len(B["passes"]))] for (E, B), (_o, base) in zip(sides[dev], pairs)]
        f, S = STT.m2_spectra(cf, MM.FS_IMU)
        _, Bs = STT.m2_spectra(cb, MM.FS_IMU)
        band_db = lambda X: {m: float(10.0 * np.log10(np.mean(X[(f >= a) & (f <= b)]))) for m, (a, b) in bands.items()}
        out["devices"][dev] = dict(
            display_db=STT.band_excess(S_, B_, bands, g), m2_db=STT.band_excess(S, Bs, bands, f),
            display_off_band_db=_off_band(g, S_, B_, bands), m2_off_band_db=_off_band(f, S, Bs, bands),
            m2_record_band_db=band_db(S), m2_twin_band_db=band_db(Bs))
    if clocks:
        out["clock_random_ms_median"] = {
            s: dict(records=float(np.median([a["random_ms"] for a in c["accuracy"] if "rigid" not in a["name"]])),
                    twins=float(np.median([a["random_ms"] for a in c["accuracy"] if "rigid" in a["name"]])))
            for s, c in clocks.items()}
    out["wall_seconds"] = time.time() - t0
    return out


def _checks(res):
    """The variants that exist elsewhere must reproduce the stored results."""
    J = json.load(open(os.path.join(TAB, f"step_timing_jitter_{STRUCT}.json")))
    T = json.load(open(os.path.join(TAB, f"step_timing_{STRUCT}.json")))["data"][SID][PLANET]["m2"]
    for r in res:
        v = r["variant"]
        if v.startswith("jit"):
            ref = [x for x in J["results"] if x["case"] == SID and x["planet"] == PLANET and abs(x["sigma_ms"] - float(v[3:])) < 1e-9][0]
            for dev in ("watch", "phone"):
                for m, x in ref["devices"][dev]["excess_db"].items():
                    assert abs(r["devices"][dev]["display_db"][m] - x) < 0.01, (v, dev, m)
        for name, var in (("jit0.0", "exact"), ("device", "device")):
            if v == name:
                for dev in ("watch", "phone"):
                    q = T[var][dev]
                    f, S, B = (np.asarray(q[k], float) for k in ("f", "S", "B"))
                    ref = STT.band_excess(S, B, r["bands"], f)
                    for m, x in ref.items():
                        assert abs(r["devices"][dev]["m2_db"][m] - x) < 0.05, (v, dev, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, VARIANTS):
            res.append(r)
            print("done", r["variant"], r.get("error", ""), flush=True)
    res.sort(key=lambda r: VARIANTS.index(r["variant"]))
    bad = [r for r in res if "error" in r]
    if bad:
        raise SystemExit(f"failed: {bad}")
    _checks(res)
    json.dump(dict(structure=STRUCT, case=SID, planet=PLANET, status="post hoc check, one walk",
                   note=__doc__.split("\n\n")[0].replace("\n", " "), variants=res, wall_seconds=time.time() - t0),
              open(os.path.join(TAB, f"device_clock_check_{STRUCT}.json"), "w"), indent=1)
    for r in res:
        print(r["variant"])
        for dev, d in r["devices"].items():
            print(f"   {dev:5s} display " + "/".join(f"{v:5.1f}" for v in d["display_db"].values())
                  + f" | m2 " + "/".join(f"{v:5.1f}" for v in d["m2_db"].values())
                  + f" | off-band {d['m2_off_band_db']:4.1f} | twin bands "
                  + "/".join(f"{v:6.1f}" for v in d["m2_twin_band_db"].values()), flush=True)


if __name__ == "__main__":
    main()
