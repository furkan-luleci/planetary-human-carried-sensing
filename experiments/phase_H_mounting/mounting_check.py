"""Phase H: sensitivity of visibility and blind recovery to how the sensor is attached.

The simulated sensors are rigidly attached to the forearm (watch) and the thigh (phone).
A real device sits on a strap or in a pocket, on soft tissue or fabric.  Here that
attachment is a linear single-degree-of-freedom mount between the body segment and the
device, the model used for skin- and strap-mounted accelerometers (Kitazaki & Griffin 1995):

    H(s) = (2 zeta w s + w^2) / (s^2 + 2 zeta w s + w^2),   w = 2 pi f_mount,

applied to each axis of the simulated specific force (the device is far lighter than the
limb, so it does not load the body and the simulation need not be re-run).  Sensor noise
is added after the mount, as in the main analysis (it is electronic).  Everything else
(spectral estimator, seeds, bands, frozen recovery rule from outputs/tables/outcomes.json)
is unchanged; while walking, the rule rejects lines near multiples of the stride
frequency of the main analysis for the same walks (a mount does not change the walk).

Mount cases (both sensors get the same mount in each case):
  f_mount 22.8 Hz and 10.7 Hz: ends of the range measured for a strap-mounted
      accelerometer at the lower back (abstract-level source, see
      docs/assumptions_and_sources.md); applied here to the wrist strap and the pocket
      without a site-specific measurement.
  f_mount 5 Hz: an assumed very loose attachment (no source), e.g. a loose pocket.
  damping ratio 0.3 and 0.1: assumed bracket (no damping value was obtained).

    python experiments/phase_H_mounting/mounting_check.py
    PLANETSENSE_STRUCTURE=S2 python experiments/phase_H_mounting/mounting_check.py   (habitat floor, Phase L)

On the habitat floor (S2, S2L) a line also counts at the mode's declared occupied frequency,
as in its main analysis; the counts at the empty frequency only are kept under "empty_only".
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_F_outcomes"))
import outcomes as oc                                            # noqa: E402
from planetsense.identification.spectral import match_peaks           # noqa: E402

MOUNTS = [dict(id="rigid", f=None, zeta=None, label="rigid (main analysis)"),
          dict(id="f22.8_z0.3", f=22.8, zeta=0.3, label="tight strap, 22.8 Hz, damping 0.3"),
          dict(id="f22.8_z0.1", f=22.8, zeta=0.1, label="tight strap, 22.8 Hz, damping 0.1"),
          dict(id="f10.7_z0.3", f=10.7, zeta=0.3, label="loose strap, 10.7 Hz, damping 0.3"),
          dict(id="f10.7_z0.1", f=10.7, zeta=0.1, label="loose strap, 10.7 Hz, damping 0.1"),
          dict(id="f5_z0.3", f=5.0, zeta=0.3, label="very loose (assumed), 5 Hz, damping 0.3"),
          dict(id="f5_z0.1", f=5.0, zeta=0.1, label="very loose (assumed), 5 Hz, damping 0.1")]
SCEN = ["stand_E1", "walk_same_E1", "walk_same_E2", "walk3d_E1", "walk3d_E2"]


def mount_H(f, f_mount, zeta):
    """Transmissibility of the mount from the body segment to the device (exact, continuous time)."""
    w = 2.0 * np.pi * f_mount
    s = 2j * np.pi * np.asarray(f, float)
    return (2.0 * zeta * w * s + w * w) / (s * s + 2.0 * zeta * w * s + w * w)


def mounted(recs, M, fs, pad_s=8.0):
    """Apply the mount to every sensor axis of each record, in the frequency domain (the
    records are long compared with the mount's impulse response; zero padding of pad_s
    seconds prevents wrap-around; the mean is removed first and restored, H(0) = 1)."""
    if M["f"] is None:
        return recs
    out = []
    for r in recs:
        F = np.asarray(r["f"], float)
        n = F.shape[0]
        N = n + int(pad_s * fs)
        mu = F.mean(axis=0)
        X = np.fft.rfft(F - mu, N, axis=0)
        H = mount_H(np.fft.rfftfreq(N, 1.0 / fs), M["f"], M["zeta"])
        r2 = dict(r)
        r2["f"] = np.fft.irfft(X * H[:, None, None], N, axis=0)[:n] + mu
        out.append(r2)
    return out


def main():
    t0 = time.time()
    struct = oc.STRUCT
    O = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json" if struct == "S1" else f"outcomes_{struct}.json")))
    occ = oc._occupied() if struct != "S1" else None
    rule = O["recovery_rule"]["settings"]
    modes = O["modes"]
    truth = [m["f"] for m in modes]
    scen = {s["id"]: s for s in oc.SCENARIOS}
    res = dict(status="SIMULATION - sensitivity check", rule=rule, modes=modes, noise_density_ug=oc.NOISE_UG,
               mounts=[], data={})
    if occ is not None:
        res.update(structure=struct, counting="empty or declared occupied frequency", occupied=occ)
    for M in MOUNTS:
        if M["f"] is not None:
            gain_db = [float(20 * np.log10(abs(h))) for h in mount_H([m["f"] for m in modes], M["f"], M["zeta"])]
        else:
            gain_db = [0.0] * len(modes)
        res["mounts"].append(dict(M, gain_at_modes_db=gain_db))
        res["data"][M["id"]] = {}
        for sid in SCEN:
            sc = scen[sid]
            res["data"][M["id"]][sid] = {}
            for p in oc.PLANETS:
                recs = oc.load_group(sc["kind"], sc["exc"], p, sc["conds"][p])
                if not recs:
                    continue
                twins = oc.load_group("walking", "rigid", p, sc["conds"][p]) if sc["kind"] == "walking" else []
                if sc["kind"] == "walking" and not twins:
                    continue
                if twins:                                # pairs, as in the main analysis
                    ok = [i for i in range(min(len(recs), len(twins))) if not recs[i]["fell"] and not twins[i]["fell"]]
                    recs, twins = [recs[i] for i in ok], [twins[i] for i in ok]
                f, Sn = oc.spectra_of(mounted(recs, M, oc.FS_IMU), sc["seg"], True, seed0=1)
                if twins:                                # same noise realization as the records
                    _, Bn = oc.spectra_of(mounted(twins, M, oc.FS_IMU), sc["seg"], True, seed0=1)
                else:                                    # standing: noise alone (the mount does not move it)
                    _, Bn = oc.noise_floor(recs, sc["seg"], seed0=1)
                d = {}
                ds = O["data"][sid][p].get("device_stride") if sc["kind"] == "walking" else None
                for key in oc.SENSORS:
                    fstr = ds[key]["f"] if ds else None
                    vis = [float(10 * np.log10(Sn[key][oc.band(f, m["f"])].mean() / Bn[key][oc.band(f, m["f"])].mean()))
                           for m in modes]
                    found, _ = oc.recover(f, Sn[key], rule, fstr)
                    found_c, _ = oc.recover(f, Bn[key], rule, fstr)
                    if occ is None:
                        mt = match_peaks(found, truth, tol_rel=0.03, tol_abs=0.15)
                        mc = match_peaks(found_c, truth, tol_rel=0.03, tol_abs=0.15)
                    else:                                # declared: empty or occupied frequency
                        mt = match_peaks(oc.snap(found, modes, occ[p]), truth, tol_rel=0.03, tol_abs=0.15)
                        mc = match_peaks(oc.snap(found_c, modes, occ[p]), truth, tol_rel=0.03, tol_abs=0.15)
                    d[key] = dict(vis_db=vis, tp=mt["tp"], fp=mt["fp"], f1=mt["f1"],
                                  found=[float(q[1]) for q in mt["pairs"]], chance_tp=mc["tp"], chance_fp=mc["fp"])
                    if occ is not None:
                        me = match_peaks(found, truth, tol_rel=0.03, tol_abs=0.15)
                        mce = match_peaks(found_c, truth, tol_rel=0.03, tol_abs=0.15)
                        d[key]["empty_only"] = dict(tp=me["tp"], fp=me["fp"], chance_tp=mce["tp"], chance_fp=mce["fp"])
                res["data"][M["id"]][sid][p] = d
                print(f"{M['id']:11s} {sid:13s} {p:5s} | thigh found {d['thigh']['tp']} (+{d['thigh']['fp']}) chance {d['thigh']['chance_tp']}"
                      f" | wrist found {d['wrist']['tp']} (+{d['wrist']['fp']}) chance {d['wrist']['chance_tp']} | {time.time() - t0:.0f} s",
                      flush=True)
    res["wall_seconds"] = time.time() - t0
    name = "mounting_check.json" if struct == "S1" else f"mounting_check_{struct}.json"
    json.dump(res, open(os.path.join(ROOT, "outputs", "tables", name), "w"), indent=1, default=float)
    print("wrote outputs/tables/" + name)


if __name__ == "__main__":
    main()
