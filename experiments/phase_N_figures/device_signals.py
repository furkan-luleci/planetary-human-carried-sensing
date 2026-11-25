"""Phase N (2026-11-22, for the signal figures): what the worn devices and the structure record,
in the time and the frequency domain, extracted from the existing records with the study's own processing.
No new simulation and no new verdict: every spectrum is computed by the functions of the main analysis
(experiments/phase_F_outcomes/outcomes.py), and the spectra after gait removal are copied unchanged from the
step-timing results (outputs/tables/step_timing_S2.json, step_timing.json).

For one structure (S2: the habitat floor, the main case; S1: the reference bridge):

* standing, shakers on (E1), Earth, Mars and the Moon, 20 records of 60 s each:
  - device spectra: outcomes.spectra_of(records, 16 s, with noise, seed0=1), i.e. per side the 3 x 3
    cross-spectral matrix of the specific force with 75 ug/sqrt(Hz) sensor noise, Welch with Hann windows
    and 50 % overlap, averaged over the records, its first singular value, mean of the left and right
    device (watch: wrists; phone: front trouser pockets);
  - background: outcomes.noise_floor(records, 16 s, seed0=1), the same noise realization alone;
  - the structure's vertical acceleration under the feet: ensemble PSD with the same segments (no noise);
* walking at 1.0 m/s, side-view walker, shakers on (E1), Earth, Mars and the Moon, 20 crossings:
  - device spectra on the structure and on rigid ground (the rigid-ground twins, same noise realization):
    outcomes.spectra_of(..., 4 s, with noise, seed0=1);
  - the structure's vertical acceleration under the walker: ensemble PSD, 4 s segments;
  - after removal of the gait's harmonics (none / device's own clock / exact clock): f, S (records) and
    B (rigid-ground twins) from outputs/tables/step_timing[_S2].json, as they are;
* time series of repetition 0 (r0), left device of each pair:
  - the structure's vertical acceleration under the person, a_under[:, 2] (m/s^2), no noise;
  - each device's specific force with the very noise realization that enters the spectra above (record 0,
    left side, seed0=1), as its magnitude |f| (m/s^2) over the analysed part of the record; the mean of |f|
    over the analysed part is stored with it (figures show |f| minus that mean, or |f| / g);
  - standing: the first 10 s of the analysed 60 s; walking: the whole analysed part of the crossing (the
    crossing minus its first 1.5 s), and the same for the rigid-ground twin r0.

Records: outputs/raw/records_S2/ (habitat floor) and outputs/raw/records/ (reference bridge and every
rigid-ground twin).  For the habitat floor and the twins used with it, the floor's span is set before the
analysed window is cut (load_group does this), exactly as in the main analysis.

Checks (written into the result and asserted): the recomputed visibility of every mode equals
outcomes[_S2].json (vis_db), the recomputed device spectra equal the stored ones (spec, four significant
digits), the RMS under the person equals deck_rms_under, and each time series carries exactly the noise that
outcomes.spectra_of adds to that record (captured from outcomes.noisy).

Writes outputs/tables/device_signals_<S>.json.

    PYTHONPATH=src python experiments/phase_N_figures/device_signals.py S2
    PYTHONPATH=src python experiments/phase_N_figures/device_signals.py S1
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_F_outcomes"))
TAB = os.path.join(ROOT, "outputs", "tables")
PLANETS = ["Earth", "Mars", "Moon"]
WALK_COND = {"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}     # side-view walker, 1.0 m/s
SHOW_STAND_S = 10.0           # seconds of a standing record shown as a time series
F_MAX = 30.0
SEED0 = 1                     # the seed of the main analysis (outcomes.py: spectra_of(..., seed0=1))


def _r(x, n=5):
    """Round to n significant digits for the JSON file."""
    return [float(f"{v:.{n}g}") for v in np.asarray(x, float)]


def device_signal(OC, rec, r_i, key, side):
    """Specific force of one device over the analysed part, with the very noise realization that
    outcomes.spectra_of(..., seed0=1) adds to record ``r_i``, device ``key`` and side ``side``."""
    m = OC.window(rec)
    sn = OC.SENSORS[key][side]
    X = np.asarray(rec["f"], float)[m][:, list(rec["sensors"]).index(sn), :]
    seed = 1000 * SEED0 + 37 * r_i + sum(map(ord, key)) + 7919 * side
    return OC.noisy(X, seed), m, sn


def check_noise(OC, rec, nper):
    """The time series must carry the very noise that outcomes.spectra_of adds to this record (index 0 of its
    group, left side): capture what spectra_of passes through outcomes.noisy and compare."""
    seen, orig = [], OC.noisy

    def spy(X, seed, noise_ug=None):
        Y = orig(X, seed, noise_ug)
        seen.append(Y)
        return Y
    OC.noisy = spy
    try:
        OC.spectra_of([rec], nper, True, seed0=SEED0)      # calls: wrist L, wrist R, thigh L, thigh R
    finally:
        OC.noisy = orig
    for key, i in (("wrist", 0), ("thigh", 2)):
        X, _, _ = device_signal(OC, rec, 0, key, 0)
        assert np.array_equal(X, seen[i]), f"noise of {rec['name']} {key} differs from the spectra's"


def time_series(OC, rec, span_s=None, r_i=0):
    """Time series of one record (r_i = its index in the pooled group, which sets its noise)."""
    t = np.asarray(rec["t"], float)
    m = OC.window(rec)
    tt = t[m] - t[m][0]
    keep = np.ones(len(tt), bool) if span_s is None else tt < span_s - 1e-9
    out = dict(t0_record_s=float(t[m][0]), t=_r(tt[keep], 6), analysed_s=float(m.sum() / OC.FS_IMU),
               g=float(rec["g"]), devices={})
    if "a_under" in rec:
        a = np.asarray(rec["a_under"], float)[m][:, 2]
        out["structure_z"] = _r(a[keep])
        out["structure_z_rms_analysed"] = float(a.std())
    if rec["activity"] == "walking":                 # where the walker is (centre of mass along the line, m)
        out["com_x"] = _r(np.asarray(rec["com"], float)[m][:, 0][keep], 4)
    for key in OC.SENSORS:
        X, _, sn = device_signal(OC, rec, r_i, key, 0)
        mag = np.linalg.norm(X, axis=1)
        out["devices"][key] = dict(sensor=sn, mag=_r(mag[keep]), mag_mean_analysed=float(mag.mean()),
                                   mag_std_analysed=float(mag.std()), mag_std_shown=float(mag[keep].std()))
    return out


def _span(struct):
    from planetsense.structure import registry as SR
    return float(SR.modal_deck(struct).params.L)


def _supports(struct):
    """Positions of the vertical supports along the walking line (m): the reference bridge is simply
    supported; the habitat floor has two equal spans (walls and a central support)."""
    L = _span(struct)
    return [0.0, L] if struct == "S1" else [0.0, 0.5 * L, L]


def structure_psd(OC, recs, nper):
    from planetsense.identification.spectral import ensemble_csd
    Ds = [np.asarray(r["a_under"], float)[OC.window(r)][:, 2:3] for r in recs]
    f, G, nseg = ensemble_csd(Ds, OC.FS_IMU, int(nper * OC.FS_IMU))
    k = f <= F_MAX
    return f[k], G[k, 0, 0].real, nseg


def vis_db(OC, f, S, B, modes):
    """The visibility measure, exactly as outcomes.py computes it."""
    out = []
    for md in modes:
        b = OC.band(f, md["f"])
        out.append(float(10 * np.log10(S[b].mean() / B[b].mean())))
    return out


def spectra_block(OC, f, S, B, modes, stored):
    """Device spectra up to 30 Hz with the checks against the main analysis's result file."""
    k = f <= F_MAX
    out = dict(f=_r(f[k], 6), devices={})
    for key in OC.SENSORS:
        v = vis_db(OC, f, S[key], B[key], modes)
        d = dict(S=_r(S[key][k]), B=_r(B[key][k]), vis_db=v)
        if stored is not None:
            d["check_vis_db_max_abs_diff"] = float(np.max(np.abs(np.array(v) - np.array(stored[key]["vis_db"]))))
            Sst = np.array(stored["spec"][key], float)
            d["check_spec_max_rel_diff"] = float(np.max(np.abs(S[key][k] - Sst) / np.maximum(Sst, 1e-300)))
        out["devices"][key] = d
    return out


def _assert_checks(d):
    """The recomputed quantities must be the main analysis's: visibility to rounding, the spectra to the
    four significant digits stored in the result file, the RMS under the person exactly."""
    for key, v in d["devices"].items():
        assert v["check_vis_db_max_abs_diff"] < 1e-6, (key, v["check_vis_db_max_abs_diff"])
        assert v["check_spec_max_rel_diff"] < 1e-3, (key, v["check_spec_max_rel_diff"])
    assert d["check_structure_rms_diff"] < 1e-9, d["check_structure_rms_diff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("struct", choices=("S2", "S1"))
    a = ap.parse_args()
    os.environ["PLANETSENSE_STRUCTURE"] = a.struct          # before outcomes.py is imported (sets its folders)
    import outcomes as OC                                   # noqa: E402  (the main analysis, used as a library)
    t0 = time.time()
    res_name = "outcomes.json" if a.struct == "S1" else f"outcomes_{a.struct}.json"
    O = json.load(open(os.path.join(TAB, res_name)))
    st_name = "step_timing.json" if a.struct == "S1" else f"step_timing_{a.struct}.json"
    ST = json.load(open(os.path.join(TAB, st_name)))
    modes = O["modes"]
    occupied = None
    if a.struct != "S1":
        occupied = json.load(open(os.path.join(TAB, "occupied_modes.json")))["data"][a.struct]
    out = dict(
        structure=a.struct, name={"S2": "habitat floor", "S1": "reference bridge"}[a.struct],
        status="SIMULATION - Mars/Moon extrapolated - not validated",
        commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
        script="experiments/phase_N_figures/device_signals.py",
        sources=dict(records=os.path.relpath(OC.REC, ROOT), twins=os.path.relpath(OC.TWIN_REC, ROOT),
                     modes=f"outputs/tables/{res_name} (modes)",
                     occupied="outputs/tables/occupied_modes.json" if occupied else None,
                     after_removal=f"outputs/tables/{st_name} (data[walk_same_E1][planet][m2][variant][device]: "
                                   "f, S, B as published)"),
        processing=dict(
            noise_ug=OC.NOISE_UG, seed0=SEED0, welch="Hann, 50 % overlap; 16 s segments standing, 4 s walking",
            device_spectrum="first singular value of the 3-axis cross-spectral matrix per side, averaged over the "
                            "records; mean of the left and right device (outcomes.spectra_of)",
            standing_background="sensor noise alone, same realization (outcomes.noise_floor)",
            walking_background="the same walk on rigid ground (rigid-ground twins), same noise realization",
            structure_psd="ensemble PSD of the structure's vertical acceleration under the person (no noise)",
            time_series="repetition 0, left device of each pair (wrist_L = watch, thigh_L = phone): magnitude of "
                        "the specific force with the noise realization of the spectra, over the analysed part; "
                        "structure: vertical acceleration under the person (no noise)",
            psd_units="(m/s^2)^2/Hz"),
        modes=modes, occupied=(occupied["occupied"] if occupied else None), span_m=_span(a.struct),
        supports_x=_supports(a.struct), stand={}, walk={})
    # ------------------------------------------------------------------ standing, shakers on
    for p in PLANETS:
        recs = OC.load_group("standing", "E1", p, None)
        f, Sn = OC.spectra_of(recs, 16.0, True, seed0=SEED0)
        _, Bn = OC.noise_floor(recs, 16.0, seed0=SEED0)
        stored = O["data"]["stand_E1"][p]
        d = spectra_block(OC, f, Sn, Bn, modes, stored)
        fs_, Ps, nseg = structure_psd(OC, recs, 16.0)
        d["structure"] = dict(S=_r(Ps), segments=int(nseg))
        rms = [float(np.asarray(r["a_under"], float)[OC.window(r)][:, 2].std()) for r in recs]
        d["structure_rms_mean"] = float(np.mean(rms))
        d["check_structure_rms_diff"] = float(abs(np.mean(rms) - stored["deck_rms_under"]["mean"]))
        d["n_records"] = len(recs)
        d["seconds"] = float(sum(OC.window(r).sum() for r in recs) / OC.FS_IMU)
        if occupied is not None:
            d["occupied_B1_from_trials"] = stored.get("occupied_B1_from_trials")
        assert recs[0]["name"].endswith("_r0")
        check_noise(OC, recs[0], 16.0)
        d["r0"] = time_series(OC, recs[0], span_s=SHOW_STAND_S, r_i=0)
        _assert_checks(d)
        out["stand"][p] = d
        print(f"standing {p}: vis " + " | ".join(f"{k} " + " ".join(f"{v:.2f}" for v in d['devices'][k]['vis_db'])
                                                 for k in OC.SENSORS)
              + f" | check vis {max(d['devices'][k]['check_vis_db_max_abs_diff'] for k in OC.SENSORS):.2e}"
              + f" spec {max(d['devices'][k]['check_spec_max_rel_diff'] for k in OC.SENSORS):.2e}"
              + f" rms {d['check_structure_rms_diff']:.1e} | {time.time() - t0:.0f} s", flush=True)
    # ------------------------------------------------------------------ walking, side-view walker, shakers on
    for p in PLANETS:
        cond = WALK_COND[p]
        recs = OC.load_group("walking", "E1", p, cond)
        twins = OC.load_group("walking", "rigid", p, cond)
        ok = [i for i in range(min(len(recs), len(twins))) if not recs[i]["fell"] and not twins[i]["fell"]]
        assert len(ok) == len(recs) == len(twins), "a fallen walker would change the pairing"
        f, Sn = OC.spectra_of(recs, 4.0, True, seed0=SEED0)
        _, Bn = OC.spectra_of(twins, 4.0, True, seed0=SEED0)
        stored = O["data"]["walk_same_E1"][p]
        d = spectra_block(OC, f, Sn, Bn, modes, stored)
        fs_, Ps, nseg = structure_psd(OC, recs, 4.0)
        d["structure"] = dict(S=_r(Ps), segments=int(nseg))
        rms = [float(np.asarray(r["a_under"], float)[OC.window(r)][:, 2].std()) for r in recs]
        d["structure_rms_mean"] = float(np.mean(rms))
        d["check_structure_rms_diff"] = float(abs(np.mean(rms) - stored["deck_rms_under"]["mean"]))
        d["n_records"] = len(recs)
        d["seconds"] = float(sum(OC.window(r).sum() for r in recs) / OC.FS_IMU)
        d["gait_on_structure"] = stored.get("gait_on_deck_trials")
        d["device_stride_hz"] = {k: v["f"] for k, v in stored.get("device_stride", {}).items()}
        assert recs[0]["name"].endswith("_r0") and twins[0]["name"].endswith("_r0")
        check_noise(OC, recs[0], 4.0)
        check_noise(OC, twins[0], 4.0)
        d["r0"] = time_series(OC, recs[0], r_i=0)
        d["r0_twin"] = time_series(OC, twins[0], r_i=0)
        d["r0_events_s"] = [float(x) - d["r0"]["t0_record_s"] for x in np.asarray(recs[0]["events"]).reshape(-1, 3)[:, 0]]
        d["r0_twin_events_s"] = [float(x) - d["r0_twin"]["t0_record_s"]
                                 for x in np.asarray(twins[0]["events"]).reshape(-1, 3)[:, 0]]
        # after removal of the gait's harmonics: the published spectra, as they are
        M2 = ST["data"]["walk_same_E1"][p]["m2"]
        d["after_removal"] = {}
        for var in ("none", "device", "exact"):
            d["after_removal"][var] = {}
            for dev, key in (("watch", "wrist"), ("phone", "thigh")):
                q = M2[var][dev]
                d["after_removal"][var][key] = dict(f=q["f"], S=q["S"], B=q["B"], f_stride=q.get("f_stride"),
                                                     no_rejection=q["no_rejection"])
        _assert_checks(d)
        out["walk"][p] = d
        print(f"walking {p}: vis " + " | ".join(f"{k} " + " ".join(f"{v:.2f}" for v in d['devices'][k]['vis_db'])
                                                for k in OC.SENSORS)
              + f" | check vis {max(d['devices'][k]['check_vis_db_max_abs_diff'] for k in OC.SENSORS):.2e}"
              + f" spec {max(d['devices'][k]['check_spec_max_rel_diff'] for k in OC.SENSORS):.2e}"
              + f" rms {d['check_structure_rms_diff']:.1e} | {time.time() - t0:.0f} s", flush=True)
    out["wall_seconds"] = time.time() - t0
    path = os.path.join(TAB, f"device_signals_{a.struct}.json")
    json.dump(out, open(path, "w"), default=float)
    print("wrote", os.path.relpath(path, ROOT), f"({os.path.getsize(path) / 1e6:.2f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
