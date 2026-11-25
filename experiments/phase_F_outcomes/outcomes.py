"""Phase F: outcomes from the time-domain trials.

For every situation (standing with shakers; walking with shakers; walking with
footsteps only), planet and worn sensor, three separate questions:

  transmission  how much of the bridge's vibration reaches the sensor, per
                bridge mode.  With shakers: force-referenced estimate (the
                part of sensor and bridge motion that is coherent with the
                known shaker forces, so the person's own movement drops out).
                With footsteps only: sensor power in excess of the same walk
                on rigid ground, relative to the bridge's power.  No sensor
                noise.
  visibility    how far each bridge frequency rises above the background in
                the worn sensor's spectrum (dB), with sensor noise; the
                background is the same activity on rigid ground with noise
                (standing on rigid ground without shakers = noise only)
  recovery      blind peak picking on the worn sensor alone.  Its settings are
                chosen on EARTH trials only and then frozen; the same rule is
                applied to Mars and Moon.  One-to-one matching to the bridge's
                natural frequencies below 30 Hz (3 % tolerance).  While walking,
                lines near multiples of the stride frequency are rejected; the
                stride frequency is the one each device estimates from its own
                signal (gait clock of Phase J, src/planetsense/evaluation/step_timing.py).
                Until Phase J it was a harmonic sum over the device's spectrum,
                which locks onto a sub-harmonic on the wrist (0.30-0.35 Hz
                instead of 0.84-1.03 Hz); see docs/phase_reports/PHASE_J_step_timing.md.

Devices are left and right pairs (watch: both wrists; phone: both trouser
pockets), combined as in the earlier study (Eq. 7): the cross-spectral matrix of each
side is averaged over the records, its first singular value taken, and the two
sides averaged.  Each record and its matched background get the same sensor-
noise realization (75 ug/sqrt(Hz)).  The analysed part of a record is the
crossing minus its first 1.5 s (walking) or the 60 s after the coupled lead-in
(standing), as in the earlier study.

The baseline study's Observability Index is computed separately, with the
baseline's own processing: experiments/phase_I_baseline_match/baseline_index.py.

    python experiments/phase_F_outcomes/outcomes.py
"""
import glob
import itertools
import json
import os
import re
import subprocess
import time

import numpy as np

from planetsense.evaluation import step_timing as ST
from planetsense.excitation.multisine import make_multisine
from planetsense.identification.spectral import ensemble_csd, match_peaks, pick_peaks, sv1
from planetsense.sensor.imu import ug_to_si, white_noise
from planetsense.simulation.records import FS_IMU, analysis_window, load_record, shaker_force

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Phase K: PLANETSENSE_STRUCTURE=S2 or S2L analyses that structure's records with S1's rigid-ground twins, S1's
# frozen recovery rule and the declared occupied frequencies (docs/phase_reports/PHASE_K_declared_rules.md)
STRUCT = os.environ.get("PLANETSENSE_STRUCTURE", "S1")
REC = os.environ.get("PLANETSENSE_REC_DIR", os.path.join(ROOT, "outputs", "raw", "records" + ("" if STRUCT == "S1" else "_" + STRUCT)))
TWIN_REC = os.path.join(ROOT, "outputs", "raw", "records") if STRUCT != "S1" else REC
PLANETS = ["Earth", "Mars", "Moon"]
SENSORS = {"wrist": ("wrist_L", "wrist_R"), "thigh": ("thigh_L", "thigh_R")}   # watch; phone in a trouser pocket
NOISE_UG = 75.0             # ug/sqrt(Hz), the earlier study, Table 1
NOISE_LEVELS = (30.0, 75.0, 300.0)
F_TRUTH_MAX = 30.0
SCENARIOS = [
    dict(id="stand_E1", label="Standing, shakers on", kind="standing", exc="E1",
         conds={"Earth": None, "Mars": None, "Moon": None}, seg=16.0,
         note="The person stands still at 4.2 m from the support; three shakers drive the structure with the same forces on every planet."),
    dict(id="walk_same_E1", label="Walking 1.0 m/s, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}, seg=4.0,
         note="Gait re-tuned on each planet to published reduced-gravity gait data; same shaker forces everywhere."),
    dict(id="walk_similar_E1", label="Earth gait slowed down, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth_1.25", "Mars": "mars_0.77_similar", "Moon": "moon_0.51_similar"}, seg=4.0,
         note="What-if case: the Earth gait replayed in slow motion (equal Froude number)."),
    dict(id="walk_same_E2", label="Walking 1.0 m/s, footsteps only", kind="walking", exc="E2",
         conds={"Earth": "earth_1.00", "Mars": "mars_1.00", "Moon": "moon_1.00"}, seg=4.0,
         note="Nothing but the walker's own footsteps moves the structure."),
    dict(id="walk_similar_E2", label="Earth gait slowed down, footsteps only", kind="walking", exc="E2",
         conds={"Earth": "earth_1.25", "Mars": "mars_0.77_similar", "Moon": "moon_0.51_similar"}, seg=4.0,
         note="What-if case, footsteps only."),
    dict(id="walk3d_E1", label="3-D walker 1.0 m/s, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth3d_1.00", "Mars": "mars3d_1.00", "Moon": "moon3d_1.00"}, seg=4.0,
         note="Phase G: the 3-D walker with sideways balance, 1.0 m/s, tuned on each planet (Stage 1 Earth, Stage 2 Mars and Moon); compare with the side-view walker."),
    dict(id="walk3d_E2", label="3-D walker 1.0 m/s, footsteps only", kind="walking", exc="E2",
         conds={"Earth": "earth3d_1.00", "Mars": "mars3d_1.00", "Moon": "moon3d_1.00"}, seg=4.0,
         note="Phase G: 3-D walker, footsteps only."),
    # the the earlier study baseline on Earth (threshold check, Phase I); not used to choose the recovery rule
    dict(id="baseline_stand", label="earlier-study baseline: standing at mid-span, shakers on", kind="standing", exc="E1",
         conds={"Earth": None, "Mars": None, "Moon": None}, seg=16.0, pat="standing_Earth-mid_E1_r*.npz",
         planets=["Earth"], note="Earth only: the person stands at mid-span, as in the baseline study."),
    dict(id="baseline_walk", label="earlier-study baseline: walking 1.5 m/s, shakers on", kind="walking", exc="E1",
         conds={"Earth": "earth_1.50", "Mars": None, "Moon": None}, seg=4.0,
         note="Earth only: walking at the baseline study's 1.5 m/s."),
]


def window(rec):
    return analysis_window(rec)


def noisy(X, seed, noise_ug=None):
    rng = np.random.default_rng(seed)
    d = ug_to_si(NOISE_UG if noise_ug is None else noise_ug)
    return X + np.column_stack([white_noise(len(X), FS_IMU, d, rng) for _ in range(X.shape[1])])


def forces(rec):
    if "F_shaker" in rec:
        return np.asarray(rec["F_shaker"], float)
    # records made before the forces were stored: regenerate (deterministic)
    ms = make_multisine(9, T=float(rec["T_sim"]) + 4.0 if "T_sim" in rec else 68.0,
                        rms=float(rec["shaker_rms"]), seed=int(rec["seed"]))
    n_sim = int(round(len(rec["t"]) * 5))
    return np.asarray(shaker_force(ms, n_sim), float)[:len(rec["t"])]


def _rep_of(path):
    return int(re.search(r"_r(\d+)\.npz$", path).group(1))


def load_group(kind, exc, planet, cond, pat=None):
    """Records of one case, in repetition order (record i pairs with twin i)."""
    if kind == "walking" and cond is None:
        return []
    pat = pat or (f"standing_{planet}_{exc}_r*.npz" if kind == "standing" else f"walking_{cond}_{exc}_r*.npz")
    d = TWIN_REC if exc == "rigid" else REC
    recs = [load_record(p) for p in sorted(glob.glob(os.path.join(d, pat)), key=_rep_of)]
    if STRUCT != "S1":                      # the S2 window: S1 twins are cut to the S2 crossing
        recs = [dict(r, deck_L=_struct_L()) for r in recs]
    return recs


def _struct_L():
    from planetsense.structure import registry as SR
    return float(SR.modal_deck(STRUCT).params.L)


def _occupied():
    """Declared occupied frequencies {planet: {label: f}} (empty frequency where none)."""
    O = json.load(open(os.path.join(ROOT, "outputs", "tables", "occupied_modes.json")))["data"][STRUCT]
    return {p: {l: (v["f"] if v else O["empty"][l]) for l, v in O["occupied"][p].items()} for p in O["occupied"]}


def snap(found, modes, occ_p):
    """A line within tolerance of a mode's occupied frequency is counted at that mode's empty frequency."""
    out = []
    for x in found:
        y = x
        for md in modes:
            fo = occ_p.get(md["label"], md["f"])
            if abs(x - fo) <= max(0.15, 0.03 * fo) and abs(x - md["f"]) > max(0.15, 0.03 * md["f"]):
                y = md["f"]
                break
        out.append(y)
    return out


def occupied_estimate(recs, f_b1, nper=16.0):
    """Standing trials: frequency of the peak of the floor's vertical acceleration under the feet near B1
    (ensemble spectrum), for the declared model-consistency check."""
    Ds = [np.asarray(r["a_under"], float)[window(r)][:, 2:3] for r in recs]
    f, G, _ = ensemble_csd(Ds, FS_IMU, int(nper * FS_IMU))
    P = G[:, 0, 0].real
    k = (f > 0.85 * f_b1) & (f < 1.15 * f_b1)
    i = np.argmax(P[k])
    j = np.nonzero(k)[0][i]
    if 0 < j < len(f) - 1:                  # parabolic refinement of the peak
        a, b, c = np.log(P[j - 1]), np.log(P[j]), np.log(P[j + 1])
        d = 0.5 * (a - c) / (a - 2 * b + c)
        return float(f[j] + d * (f[1] - f[0]))
    return float(f[j])


def spectra_of(recs, nper, with_noise, seed0=0, noise_ug=None, zero=False):
    """Ensemble first-singular-value spectrum of each device (3 accelerometer
    axes per side; the earlier study, Eq. 7: CSD averaged over records, first singular
    value per side, mean of left and right).  Noise: record i, device ``key``,
    side ``s`` always gets the same realization for a given ``seed0``, so a
    background computed with the same ``seed0`` (the rigid-ground twins, or
    ``zero=True``: the records with their signal removed) carries the very same
    noise.  ``zero=True`` keeps only the lengths of the records."""
    out = {}
    f = None
    for key, sns in SENSORS.items():
        acc = None
        for side, sn in enumerate(sns):
            Xs = []
            for r_i, r in enumerate(recs):
                m = window(r)
                X = np.asarray(r["f"], float)[m][:, list(r["sensors"]).index(sn), :]
                if zero:
                    X = np.zeros_like(X)
                if with_noise:
                    X = noisy(X, 1000 * seed0 + 37 * r_i + sum(map(ord, key)) + 7919 * side, noise_ug)
                Xs.append(X)
            f, G, _ = ensemble_csd(Xs, FS_IMU, int(nper * FS_IMU))
            acc = sv1(G) if acc is None else acc + sv1(G)
        out[key] = acc / len(sns)
    return f, out


def noise_floor(recs, nper, seed0=0):
    """Sensor noise alone, with the realization the records themselves carry for this ``seed0``."""
    return spectra_of(recs, nper, True, seed0=seed0, zero=True)


def band(f, f0, frac=0.03, lo=0.15):
    return np.abs(f - f0) <= max(lo, frac * f0)


def transmission_forced(recs, nper, modes):
    """Force-referenced: coherent output power of sensor (3 axes) and of the
    bridge under the person (vertical), per mode band; ratio -> amplitude ratio."""
    res = {k: [] for k in SENSORS}
    num = {k: None for k in SENSORS}
    den = None
    f = None
    for key, sns in SENSORS.items():
        for sn in sns:                                   # both sides; their coherent powers are averaged
            Xs = []
            for r in recs:
                m = window(r)
                F = forces(r)[:len(r["t"])][m]
                Y = np.asarray(r["f"], float)[m][:, list(r["sensors"]).index(sn), :]
                D = np.asarray(r["a_under"], float)[m][:, 2:3]
                Xs.append(np.hstack([F, Y, D]))
            f, G, _ = ensemble_csd(Xs, FS_IMU, int(nper * FS_IMU))
            Gff, Gyf = G[:, :9, :9], G[:, 9:, :9]
            # coherent output power  Gyf Gff^-1 Gfy  (diagonal) per output channel
            P = np.zeros((len(f), 4))
            for i in range(len(f)):
                Ainv = np.linalg.pinv(Gff[i], rcond=1e-10)
                P[i] = np.real(np.einsum("ij,jk,ik->i", Gyf[i], Ainv, np.conj(Gyf[i])))
            ps = P[:, :3].sum(axis=1) / len(sns)
            num[key] = ps if num[key] is None else num[key] + ps
            den = P[:, 3]
    for key in SENSORS:
        for md in modes:
            b = band(f, md["f"])
            res[key].append(float(np.sqrt(np.trapezoid(num[key][b], f[b]) / max(np.trapezoid(den[b], f[b]), 1e-300))))
    return res


def transmission_excess(recs, twins, nper, modes):
    """Footsteps only: sensor power in excess of the rigid-ground twin, relative
    to the bridge's vertical power under the person, per mode band."""
    f, Sd = spectra_of(recs, nper, False)
    _, Sr = spectra_of(twins, nper, False)
    Ds = [np.asarray(r["a_under"], float)[window(r)][:, 2:3] for r in recs]
    _, Gd, _ = ensemble_csd(Ds, FS_IMU, int(nper * FS_IMU))
    D = Gd[:, 0, 0].real
    out = {}
    for key in SENSORS:
        out[key] = []
        for md in modes:
            b = band(f, md["f"])
            ex = max(np.trapezoid(Sd[key][b] - Sr[key][b], f[b]), 0.0)
            out[key].append(float(np.sqrt(ex / max(np.trapezoid(D[b], f[b]), 1e-300))))
    return out


def stride_frequency(f, S, lo=0.3, hi=1.3, nh=12):
    """Harmonic-sum estimate of the stride frequency from the sensor spectrum
    alone (gait lines sit at multiples of it).  Uses no knowledge of the bridge.
    Superseded in Phase J: on the wrist it locks onto a sub-harmonic.  Kept to
    report what the rule used before (field ``stride_harmonic_sum``)."""
    L = np.log(np.maximum(S, 1e-300))
    best, fbest = -np.inf, np.nan
    for f0 in np.arange(lo, hi, 0.005):
        k = np.arange(1, nh + 1) * f0
        k = k[k < f[-1]]
        v = np.interp(k, f, L).mean()
        if v > best:
            best, fbest = v, f0
    return float(fbest)


def device_stride(recs, twins):
    """Stride frequency each device estimates from its own signal (Phase J gait clock):
    per side, events per stride and stride period pooled over the records and their
    rigid-ground twins; the device's value is the mean of its two sides.  Also each
    walk's own value (for the per-crossing scores)."""
    allr = list(recs) + list(twins)
    out = {}
    for key, sns in SENSORS.items():
        ps = [ST.pooled_stride(allr, sn, NOISE_UG) for sn in sns]
        per = [float(np.mean([1.0 / q["per_walk_stride_s"][i] for q in ps])) for i in range(len(allr))]
        out[key] = dict(f=float(np.mean([1.0 / q["stride_s"] for q in ps])), records=per[:len(recs)],
                        twins=per[len(recs):], events_per_stride=[q["n_e"] for q in ps])
    return out


def recover(f, S, settings, f_stride):
    """Blind peak picking; ``f_stride`` is the device's stride frequency while walking
    (lines near its multiples are rejected) and None when standing."""
    peaks = pick_peaks(f, S, fmin=1.0, fmax=F_TRUTH_MAX, prominence_db=settings["prom"],
                       background_hz=settings["bg"], harmonics_of=f_stride,
                       harmonic_halfwidth=settings["hw"], max_harmonic_hz=settings["hmax"])
    return [p[0] for p in peaks], f_stride


def main():
    t_start = time.time()
    if STRUCT == "S1":
        deck_modes = json.load(open(os.path.join(ROOT, "outputs", "tables", "standing_v2_summary.json")))["deck_modes"]
    else:
        from planetsense.structure import registry as SR
        md_ = SR.modal_deck(STRUCT)
        deck_modes = [dict(label=l, f=float(q)) for l, q in zip(md_.labels, md_.freq_hz)]
    modes = [m for m in deck_modes if m["f"] < F_TRUTH_MAX]
    occ = _occupied() if STRUCT != "S1" else None
    truth = [m["f"] for m in modes]
    out = dict(status="SIMULATION - Mars/Moon extrapolated - not validated", noise_density_ug=NOISE_UG, structure=STRUCT,
               commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
               modes=[dict(label=m["label"], f=m["f"]) for m in modes], scenarios=[], data={})
    cache, bg_cache = {}, {}
    for sc in SCENARIOS:
        have = {}
        for p in sc.get("planets", PLANETS):
            recs = load_group(sc["kind"], sc["exc"], p, sc["conds"][p], pat=sc.get("pat"))
            twins = load_group("walking", "rigid", p, sc["conds"][p]) if sc["kind"] == "walking" else []
            if recs and (sc["kind"] == "standing" or twins):
                have[p] = (recs, twins)
        if not have:
            continue
        entry = {k: sc[k] for k in ("id", "label", "note")}
        if STRUCT != "S1" and sc["id"] == "stand_E1":     # the note above describes the reference bridge's standing point
            from planetsense.structure import registry as SR
            entry["note"] = (f"The person stands still {SR.get(STRUCT).stand_xy[0]:.2f} m from the wall on the floor's beam line; "
                             "three shakers drive the structure with the same forces on every planet.")
        out["scenarios"].append(entry)
        out["data"][sc["id"]] = {}
        for p, (recs, twins) in have.items():
            nper = sc["seg"]
            n_fell = int(sum(bool(r["fell"]) for r in recs)) + int(sum(bool(r["fell"]) for r in twins))
            if twins:                                    # pairs: a fallen walker removes its pair
                ok = [i for i in range(min(len(recs), len(twins))) if not recs[i]["fell"] and not twins[i]["fell"]]
                recs, twins = [recs[i] for i in ok], [twins[i] for i in ok]
            f, Sn = spectra_of(recs, nper, True, seed0=1)
            if twins:                                    # rigid-ground twins, same noise realization
                _, Bn = spectra_of(twins, nper, True, seed0=1)
            else:                                        # standing: noise alone, same realization
                _, Bn = noise_floor(recs, nper, seed0=1)
            # Transmission is estimated for standing only.  While walking, the bridge-induced
            # part of the sensor signal is smaller than the gait's own variability: with nine
            # force references and ~30 averages the force-coherent estimate is biased by gait
            # leakage, and paired shaker-on/off trials drift apart by up to 2 ms in footfall
            # timing, whose effect exceeds the bridge's (see docs/phase_reports/PHASE_F_outcomes.md).
            if sc["kind"] == "standing" and sc["exc"] == "E1":
                tr = transmission_forced(recs, nper, modes)
            else:
                tr = {k: None for k in SENSORS}
            d = dict(n_records=len(recs), seconds=float(sum(window(r).sum() for r in recs) / FS_IMU),
                     spec=dict(f=[round(float(v), 4) for v in f[f <= 30.0]]))
            # bridge vibration level: vertical RMS under the person and at mid-span (on-deck window)
            ru = [float(np.asarray(r["a_under"], float)[window(r)][:, 2].std()) for r in recs]
            rm = [float(np.asarray(r["a_ref"], float)[window(r)][:, 1, 2].std()) for r in recs]
            d["deck_rms_under"] = dict(mean=float(np.mean(ru)), sd=float(np.std(ru)), values=ru)
            d["deck_rms_midspan"] = dict(mean=float(np.mean(rm)), sd=float(np.std(rm)), values=rm)
            if sc["kind"] == "walking":
                gs = [r["gait"] for r in recs if isinstance(r.get("gait"), dict)]
                d["gait_on_deck_trials"] = {k: float(np.mean([g_[k] for g_ in gs if k in g_]))
                                            for k in ("speed", "step_frequency", "grf_foot_peak_over_W", "duty_factor")
                                            if any(k in g_ for g_ in gs)}
                d["falls"] = n_fell                     # in the records and their twins (pairs left out)
            strides = device_stride(recs, twins) if sc["kind"] == "walking" else None
            if strides:
                d["device_stride"] = {k: dict(f=v["f"], events_per_stride=v["events_per_stride"],
                                              harmonic_sum=stride_frequency(f, Sn[k])) for k, v in strides.items()}
            keep = f <= 30.0
            for key in SENSORS:
                d["spec"][key] = [float(f"{v:.4g}") for v in Sn[key][keep]]
                vis = []
                for md in modes:
                    b = band(f, md["f"])
                    vis.append(float(10 * np.log10(Sn[key][b].mean() / Bn[key][b].mean())))
                d[key] = dict(trans=tr[key], vis_db=vis)
            if occ is not None and sc["kind"] == "standing":
                d["occupied_B1_from_trials"] = occupied_estimate(recs, modes[0]["f"])
                d["occupied_B1_predicted"] = occ[p][modes[0]["label"]]
            cache[(sc["id"], p)] = (f, Sn, recs, sc, strides)
            bg_cache[(sc["id"], p)] = Bn
            out["data"][sc["id"]][p] = d
            print(f"{sc['id']:16s} {p:5s} records {len(recs)} ({d['seconds']:.0f} s) | vis thigh "
                  + " ".join(f"{v:.1f}" for v in d["thigh"]["vis_db"]) + f" | {time.time() - t_start:.0f} s", flush=True)
    # ---------------------------------------------- recovery rule, tuned on Earth only
    grid = [dict(prom=a, bg=b, hw=c, hmax=d) for a, b, c, d in itertools.product(
        (4.0, 6.0, 8.0, 10.0), (2.0, 4.0), (0.15, 0.3), (10.0, 20.0, 30.0))]

    def per_rep(sc_id, p, key, st, alt=False):
        # alt (habitat floors only): a line also counts at the declared occupied frequency
        f, Sn, recs, sc, strides = cache[(sc_id, p)]
        scores = []
        for i, r in enumerate(recs):
            fr, S1 = spectra_of([r], sc["seg"], True, seed0=7 + i)
            found, _ = recover(fr, S1[key], st, strides[key]["records"][i] if strides else None)
            scores.append(match_peaks(snap(found, modes, occ[p]) if alt else found, truth)["f1"])
        return scores

    # chosen on the Earth cases of the main study only; the 3-D walker (Phase G) is scored with the frozen rule
    earth = [(sid, "Earth") for (sid, pl) in cache if pl == "Earth" and not sid.startswith(("walk3d", "baseline"))]
    best = None
    if STRUCT != "S1":                       # declared: S1's frozen rule, not re-tuned
        S1O = json.load(open(os.path.join(ROOT, "outputs", "tables", "outcomes.json")))
        best, grid = (float("nan"), S1O["recovery_rule"]["settings"]), []
    for st in grid:
        sc_list = []
        for sid, p in earth:
            f, Sn, recs, sc, strides = cache[(sid, p)]
            for key in SENSORS:
                found, _ = recover(f, Sn[key], st, strides[key]["f"] if strides else None)
                sc_list.append(match_peaks(found, truth)["f1"])
        v = float(np.mean(sc_list))
        if best is None or v > best[0] + 1e-12:
            best = (v, st)
    frozen = best[1]
    out["recovery_rule"] = dict(settings=frozen, earth_mean_f1=(None if best[0] != best[0] else best[0]), grid_size=len(grid),
                                chosen_on="Earth trials only (all situations and sensors), then frozen",
                                stride_frequency="each device's own gait clock (Phase J): pooled over the records and "
                                                 "twins of a case per side, mean of the sides; per crossing for the "
                                                 "per-crossing scores",
                                before_phase_J=dict(settings={"prom": 4.0, "bg": 4.0, "hw": 0.15, "hmax": 20.0},
                                                    stride_frequency="harmonic sum over the device spectrum"),
                                tolerance="max(0.15 Hz, 3 %)", truth="bridge natural frequencies below 30 Hz")
    for (sid, p), (f, Sn, recs, sc, strides) in cache.items():
        for key in SENSORS:
            fstr = strides[key]["f"] if strides else None
            found, fs = recover(f, Sn[key], frozen, fstr)
            m = match_peaks(found, truth, tol_rel=0.03, tol_abs=0.15)
            fa = [x for x in found if all(abs(x - q[1]) > 1e-9 for q in m["pairs"])]
            out["data"][sid][p][key]["rec"] = dict(m, truth=truth, false_alarms=sorted(fa), stride_frequency=fs,
                                                  f1_per_record=per_rep(sid, p, key, frozen))
            # chance level: the same frozen rule on the background spectrum, where no bridge exists
            # (walking: the rigid-ground twin with noise; standing: sensor noise only)
            found_c, _ = recover(f, bg_cache[(sid, p)][key], frozen, fstr)
            mc = match_peaks(found_c, truth, tol_rel=0.03, tol_abs=0.15)
            out["data"][sid][p][key]["rec_chance"] = dict(tp=mc["tp"], fp=mc["fp"], fn=mc["fn"], recall=mc["recall"],
                                                          precision=mc["precision"], f1=mc["f1"])
            if occ is not None:                  # declared: empty or occupied frequency
                ma = match_peaks(snap(found, modes, occ[p]), truth, tol_rel=0.03, tol_abs=0.15)
                mca = match_peaks(snap(found_c, modes, occ[p]), truth, tol_rel=0.03, tol_abs=0.15)
                out["data"][sid][p][key]["rec_alt"] = dict(tp=ma["tp"], fp=ma["fp"], f1=ma["f1"],
                                                           matched=[q[0] for q in ma["pairs"]],
                                                           f1_per_record=per_rep(sid, p, key, frozen, alt=True))
                out["data"][sid][p][key]["rec_alt_chance"] = dict(tp=mca["tp"], fp=mca["fp"], f1=mca["f1"])
    # ---------------------------------------- sensitivity of recovery (frozen rule)
    # (a) sensor noise density; (b) number of crossings pooled (walking) or record length (standing)
    for (sid, p), (f, Sn, recs, sc, strides) in cache.items():
        for key in SENSORS:
            fstr = strides[key]["f"] if strides else None
            d = out["data"][sid][p][key]
            d["rec_vs_noise"] = {}
            for nu in NOISE_LEVELS:
                fr, S1 = spectra_of(recs, sc["seg"], True, seed0=3, noise_ug=nu)
                found, _ = recover(fr, S1[key], frozen, fstr)
                m = match_peaks(found, truth)
                d["rec_vs_noise"][f"{nu:g}"] = dict(tp=m["tp"], fp=m["fp"], f1=m["f1"])
                if occ is not None:              # declared: empty or occupied frequency
                    ma = match_peaks(snap(found, modes, occ[p]), truth)
                    d["rec_vs_noise"][f"{nu:g}"].update(tp_alt=ma["tp"], fp_alt=ma["fp"], f1_alt=ma["f1"])
            d["rec_vs_n"] = {}
            for n in range(1, len(recs) + 1):
                fr, S1 = spectra_of(recs[:n], sc["seg"], True, seed0=4)
                found, _ = recover(fr, S1[key], frozen, fstr)
                m = match_peaks(found, truth)
                d["rec_vs_n"][str(n)] = dict(tp=m["tp"], fp=m["fp"], f1=m["f1"],
                                             seconds=float(sum(window(r).sum() for r in recs[:n]) / FS_IMU))
                if occ is not None:              # declared: empty or occupied frequency
                    ma = match_peaks(snap(found, modes, occ[p]), truth)
                    d["rec_vs_n"][str(n)].update(tp_alt=ma["tp"], fp_alt=ma["fp"], f1_alt=ma["f1"])
    out["wall_seconds"] = time.time() - t_start
    if STRUCT != "S1":
        out["recovery_rule"]["chosen_on"] = "S1's frozen rule, not re-tuned (declared, Phase K)"
        out["occupied"] = occ
    default_out = os.path.join(ROOT, "outputs", "tables", "outcomes.json" if STRUCT == "S1" else f"outcomes_{STRUCT}.json")
    json.dump(out, open(os.environ.get("PLANETSENSE_OUTCOMES_OUT", default_out), "w"),
              indent=1, default=float)
    print("frozen recovery rule:", frozen, "Earth mean F1 %.3f" % best[0])
    print("wrote", os.environ.get("PLANETSENSE_OUTCOMES_OUT", default_out))


if __name__ == "__main__":
    main()
