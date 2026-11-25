"""Phase M (2026-11-22, post hoc): would longer, continuous walking
records (back-and-forth crossings without stepping off) improve the walking results?  Two checks
on the existing records; nothing new is simulated and no published output is changed.

Check 1, spectral window.  The main analysis cuts each crossing into 4 s windows (50 % overlap);
the earlier study's processing uses one window per crossing.  Blind recovery is recomputed with one
window spanning the shortest crossing of the case (records and twins), for
  (a) the main blind analysis (experiments/phase_F_outcomes/outcomes.py: frozen rule, harmonic
      rejection at the stride frequency of each device's own clock; criterion C2), and
  (b) blind recovery after gait-harmonic removal (the M2 path of experiments/phase_J_step_timing and
      experiments/phase_K_S2: no removal, exact clock, device clock; harmonic rejection off; C3).

Check 2, length of the harmonic fit.  The removal (vendored from the earlier study) fits a sine and
a cosine at each detected gait line over the whole analysed crossing and subtracts the fit.  Here
the same lines and clocks are fitted over a quarter, a half and the whole crossing (reimplemented;
the whole-crossing fit must reproduce the vendored output), and over all crossings of the case at
once, after each record's clock is aligned to the others by the device itself (integer event
shifts chosen from the record's own harmonic phases).  The pooled fit stands in, optimistically, for
minutes of continuous walking with a perfectly steady gait: no turns, no speed changes.

Guards: the 4 s results of the published paths are recomputed and must equal the published counts
(outcomes*.json, step_timing*.json).

Writes outputs/tables/walking_checks_<S>.json.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/phase_M_walking_checks/walking_checks.py S2 [--workers 4]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("phase_J_step_timing", "phase_I_baseline_match", "phase_K_S2", "phase_F_outcomes"):
    sys.path.insert(0, os.path.join(ROOT, "experiments", _sub))
sys.path.insert(0, os.path.join(ROOT, "src"))
TAB = os.path.join(ROOT, "outputs", "tables")
WALK4 = ("walk_same_E1", "walk_same_E2", "walk3d_E1", "walk3d_E2")      # the 24 cases compared between structures
FITS = (("quarter", 4), ("half", 2), ("whole", 1))


def T(name):
    return os.path.join(TAB, name)


def tasks(struct):
    if struct == "S1":
        import baseline_index as MI
        return [dict(tk, struct="S1") for tk in MI.tasks() if tk["kind"] == "walking"]
    import step_timing_s2 as S2T
    return S2T.tasks(struct)


def fit_remove(sig, tb, lines, n_win=1):
    """Least-squares sine and cosine at each line (Hz on the clock tb), subtracted, in n_win equal
    windows.  Columns are capped at 15 % of the window's samples, highest lines dropped first, as in
    the vendored removal (legacy_pipeline._remove_gait_harmonics)."""
    out = np.asarray(sig, float).copy()
    edges = np.linspace(0, len(out), n_win + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        cols = []
        for c in sorted(lines):
            cols += [np.sin(2 * np.pi * c * tb[a:b]), np.cos(2 * np.pi * c * tb[a:b])]
        lim = max(4, int(0.15 * (b - a)))
        while len(cols) > lim:
            del cols[-2:]
        if not cols:
            continue
        X = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(X, out[a:b], rcond=None)
        out[a:b] -= X @ coef
    return out


def _coef(sig, tb, lines):
    """Complex amplitude z = c_cos - i c_sin of each line, from a whole-record fit."""
    cols = []
    for c in lines:
        cols += [np.sin(2 * np.pi * c * tb), np.cos(2 * np.pi * c * tb)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), sig, rcond=None)
    return coef[1::2] - 1j * coef[0::2]


def aligned_clocks(chans, taus, fp, n_e, lines):
    """Shift each record's absolute clock by a whole number of events (0..n_e-1) so that its gait
    harmonics line up with the others' (two passes against the running mean).  Uses only the
    device's own signal.  Returns the shifted clocks, the shifts and the phase coherence of the
    lines after alignment (|mean z| / mean |z|, amplitude-weighted over lines and axes)."""
    k = np.round(2.0 * np.asarray(lines) / fp).astype(int)                 # harmonic order of each line
    Z = [np.concatenate([_coef(ch[a], tb, lines) for a in range(3)]) for ch, tb in zip(chans, taus)]
    K = np.concatenate([k, k, k])
    rot = lambda z, s: z * np.exp(-2j * np.pi * K * s / n_e)
    shifts = [0] * len(Z)
    for _ in range(2):
        ref = np.mean([rot(z, s) for z, s in zip(Z, shifts)], axis=0)
        shifts = [int(np.argmax([np.real(np.vdot(ref, rot(z, s))) for s in range(n_e)])) for z in Z]
    A = np.array([rot(z, s) for z, s in zip(Z, shifts)])
    w = np.abs(A).mean(axis=0)
    coh = float(np.sum(np.abs(A.mean(axis=0))) / np.sum(w)) if np.sum(w) > 0 else float("nan")
    return [tb + 2.0 * s / (n_e * fp) for tb, s in zip(taus, shifts)], shifts, coh


def pooled_remove(chans, taus, lines_by_axis):
    """One least-squares fit of the gait lines over all records at once (aligned clocks)."""
    out = [np.array(ch, float).copy() for ch in chans]
    for a in range(3):
        lines = lines_by_axis[a]
        if not lines:
            continue
        X = np.vstack([np.column_stack([f(2 * np.pi * c * tb) for c in lines for f in (np.sin, np.cos)]) for tb in taus])
        y = np.concatenate([ch[a] for ch in chans])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r0 = 0
        for i, tb in enumerate(taus):
            n = len(tb)
            out[i][a] -= X[r0:r0 + n] @ coef
            r0 += n
    return out


def run_task(tk):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import outcomes as OC                       # reads PLANETSENSE_STRUCTURE, set by main()
    import step_timing_test as STT
    import step_timing_s2 as S2T
    from planetsense.evaluation import legacy_pipeline as LP
    from planetsense.evaluation import baseline_match as MM
    from planetsense.identification.spectral import match_peaks, pick_peaks
    from planetsense.simulation.records import load_record
    t0 = time.time()
    struct, sid, planet = tk["struct"], tk["sid"], tk["planet"]
    alt = struct != "S1"
    O = json.load(open(T("outcomes.json" if struct == "S1" else f"outcomes_{struct}.json")))
    rule = json.load(open(T("outcomes.json")))["recovery_rule"]["settings"]
    modes = O["modes"]
    truth = [m["f"] for m in modes]
    occ_p = OC._occupied()[planet] if alt else None
    L = OC._struct_L() if alt else None
    recs = [load_record(f) for f in tk["files"]]
    twins = [load_record(f) for f in tk["twins"]]
    if alt:
        recs, twins = [dict(r, deck_L=L) for r in recs], [dict(r, deck_L=L) for r in twins]
    keep = [i for i in range(len(recs)) if not recs[i].get("fell") and not twins[i].get("fell")]
    recs, twins = [recs[i] for i in keep], [twins[i] for i in keep]
    fs = float(MM.FS_IMU)

    def count(f, S, B, f_stride):
        pk = [[q[0] for q in pick_peaks(f, spec, fmin=1.0, fmax=30.0, prominence_db=rule["prom"], background_hz=rule["bg"],
                                       harmonics_of=f_stride, harmonic_halfwidth=rule["hw"], max_harmonic_hz=rule["hmax"])]
              for spec in (S, B)]
        if alt:
            pk = [S2T.snap(x, modes, occ_p) for x in pk]
        m = [match_peaks(x, truth, tol_rel=0.03, tol_abs=0.15) for x in pk]
        return dict(tp=int(m[0]["tp"]), fp=int(m[0]["fp"]), chance=int(m[1]["tp"]), chance_fp=int(m[1]["fp"]))

    def excess(f, S, B):
        out = {}
        for md in modes:
            fe = [md["f"]] + ([occ_p[md["label"]]] if alt and occ_p.get(md["label"]) else [])
            k = np.zeros(len(f), bool)
            for f0 in fe:
                k |= np.abs(f - f0) <= max(0.15, 0.03 * f0)
            out[md["label"]] = float(10 * np.log10(S[k].mean() / B[k].mean())) if k.sum() else None
        return out

    res = dict(struct=struct, sid=sid, planet=planet, cond=tk["cond"], n_records=len(recs))
    # ------------------------------------------------ check 1a: the main blind analysis (C2)
    dpub = O["data"].get(sid, {}).get(planet)
    if dpub is not None:
        n_one = int(min(OC.window(r).sum() for r in recs + twins))
        rk, ck = ("rec_alt", "rec_alt_chance") if alt else ("rec", "rec_chance")
        res["main"] = dict(window_s=dict(w4=4.0, crossing=n_one / fs),
                           published={k: dict(tp=dpub[k][rk]["tp"], chance=dpub[k][ck]["tp"]) for k in OC.SENSORS})
        for lab, nper in (("w4", 4.0), ("crossing", n_one / fs)):
            f, Sn = OC.spectra_of(recs, nper, True, seed0=1)
            _, Bn = OC.spectra_of(twins, nper, True, seed0=1)
            res["main"][lab] = {k: dict(count(f, Sn[k], Bn[k], dpub["device_stride"][k]["f"]), vis=excess(f, Sn[k], Bn[k]))
                                for k in OC.SENSORS}
    # ------------------------------------------------ checks 1b and 2: after gait-harmonic removal (M2)
    STp = json.load(open(T("step_timing.json" if struct == "S1" else f"step_timing_{struct}.json")))
    pub_m2 = STp["data"][sid][planet]["m2"]
    mv = json.load(open(os.path.join(ROOT, "outputs", "gait", tk["cond"] + ".json")))["metrics_validation_30s"]
    fp_nom = float(mv["step_frequency"])
    clocks = STT.device_clocks(recs, twins)
    variants = {}                                # name -> {dev: (list per side of rec chans, of twin chans)}
    diag = dict(reimpl_max_rel_diff=0.0, coherence={}, shifts={})
    lens = []
    for clock in ("exact", "device"):
        sides = {}
        for dev, pairs in STT.SENSORS.items():
            sides[dev] = []
            for ours, base in pairs:
                if clock == "exact":
                    E = MM.ensemble(recs, False, fp=fp_nom, sensors=[base])
                    B = MM.ensemble(twins, False, fp=fp_nom, sensors=[base])
                else:
                    c = clocks[ours]
                    E = MM.ensemble(recs, False, fp=c["fp"], sensors=[base], clocks=c["clocks_rec"])
                    B = MM.ensemble(twins, False, fp=c["fp"], sensors=[base], clocks=c["clocks_twin"])
                sides[dev].append((E, B, ours, base))
        with STT._Capture(LP) as cap:
            MM.device_curves_sided({d: [(E, B) for E, B, _o, _b in v] for d, v in sides.items()})
        for dev, pairs in sides.items():
            part = MM.DEVICES[dev][0]
            for E, B, ours, base in pairs:
                with MM._pairing():
                    lines = LP.matched_comb_lines(E, B, part)
                n_e = clocks[ours]["n_e"] if clock == "device" else 2
                fpc = clocks[ours]["fp"] if clock == "device" else fp_nom
                for role, ens in (("rec", E), ("twin", B)):
                    raw = STT.raw_channels(LP, ens, base)
                    taus = []
                    for i, p in enumerate(ens["passes"]):
                        i0 = LP._trim_index(p, len(p["clean"][base]))
                        taus.append(np.asarray(p["tau"], float)[i0:])
                    lens += [ch.shape[1] for ch in raw]
                    pub = [cap.store[(id(ens), i, base)] for i in range(len(raw))]
                    if clock == "exact":
                        variants.setdefault("none", {}).setdefault(dev, {}).setdefault(role, []).append(raw)
                    variants.setdefault(f"{clock}:published", {}).setdefault(dev, {}).setdefault(role, []).append(pub)
                    for name, nw in FITS:
                        rm = [np.array([fit_remove(ch[a], tb - tb[0], lines[(i, base, ax)], nw)
                                        for a, ax in enumerate(LP.AXIS_NAMES)]) for i, (ch, tb) in enumerate(zip(raw, taus))]
                        if name == "whole":
                            for x, y in zip(rm, pub):
                                diag["reimpl_max_rel_diff"] = max(diag["reimpl_max_rel_diff"],
                                                                  float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30)))
                        variants.setdefault(f"{clock}:{name}", {}).setdefault(dev, {}).setdefault(role, []).append(rm)
                    # pooled over the case: union of each axis's lines over records and twins
                    lba = []
                    for ax in LP.AXIS_NAMES:
                        u = set()
                        for (i, s_, a_), v in lines.items():
                            if s_ == base and a_ == ax:
                                u |= set(round(float(c), 9) for c in v)
                        lba.append(sorted(u))
                    alines = sorted(set(lba[0]) | set(lba[1]) | set(lba[2]))
                    tal, sh, coh = aligned_clocks(raw, taus, fpc, n_e, alines) if alines else (taus, [0] * len(taus), float("nan"))
                    diag["coherence"][f"{clock}:{base}:{role}"] = coh
                    diag["shifts"][f"{clock}:{base}:{role}"] = sh
                    variants.setdefault(f"{clock}:pooled", {}).setdefault(dev, {}).setdefault(role, []).append(
                        pooled_remove(raw, tal, lba))
    n_one = int(min(lens))
    res["m2_window_s"] = dict(w4=4.0, crossing=n_one / fs)
    res["m2"] = {}
    for vname, byd in variants.items():
        res["m2"][vname] = {}
        for dev, d in byd.items():
            fstr = float(np.mean([1.0 / clocks[o]["stride_s"] for o, _b in STT.SENSORS[dev]]))
            res["m2"][vname][dev] = {}
            for lab, nper in (("w4", 4.0), ("crossing", n_one / fs)):
                f, S = STT.m2_spectra(d["rec"], fs, nper)
                _, Bs = STT.m2_spectra(d["twin"], fs, nper)
                res["m2"][vname][dev][lab] = dict(count(f, S, Bs, None), vis=excess(f, S, Bs))
    # guard: the published M2 counts (4 s windows, harmonic rejection off)
    res["m2_published"] = {v: {dev: dict(tp=pub_m2[v][dev]["no_rejection"]["tp"], chance=pub_m2[v][dev]["no_rejection"]["chance_tp"])
                               for dev in ("watch", "phone")} for v in ("none", "exact", "device") if v in pub_m2}
    res["diag"] = diag
    res["wall"] = time.time() - t0
    return res


def summarize(results):
    def above(c):
        return c["tp"] > c["chance"] + 1
    S = {}
    sets = {"compared_24": [r for r in results if r["sid"] in WALK4], "all": results}
    for setname, rs in sets.items():
        d = {}
        if all("main" in r for r in rs):
            for lab in ("w4", "crossing"):
                d[f"main:{lab}"] = sum(above(r["main"][lab][k]) for r in rs for k in ("wrist", "thigh"))
            d["main:published"] = sum(above(r["main"]["published"][k]) for r in rs for k in ("wrist", "thigh"))
        for v in rs[0]["m2"]:
            for lab in ("w4", "crossing"):
                d[f"{v}:{lab}"] = sum(above(r["m2"][v][dev][lab]) for r in rs for dev in ("watch", "phone"))
        for v in ("none", "exact", "device"):
            d[f"{v}:published_counts"] = sum(above(r["m2_published"][v][dev]) for r in rs for dev in ("watch", "phone")
                                             if v in r["m2_published"])
        d["n_cases"] = 2 * len(rs)
        S[setname] = d
    # guards
    g = dict(main_matches_published=all(r["main"]["w4"][k]["tp"] == r["main"]["published"][k]["tp"]
                                        and r["main"]["w4"][k]["chance"] == r["main"]["published"][k]["chance"]
                                        for r in results if "main" in r for k in ("wrist", "thigh")),
             m2_matches_published=all(r["m2"][f"{v}:published" if v != "none" else "none"][dev]["w4"]["tp"] == r["m2_published"][v][dev]["tp"]
                                      and r["m2"][f"{v}:published" if v != "none" else "none"][dev]["w4"]["chance"] == r["m2_published"][v][dev]["chance"]
                                      for r in results for v in ("none", "exact", "device") if v in r["m2_published"]
                                      for dev in ("watch", "phone")),
             reimpl_max_rel_diff=max(r["diag"]["reimpl_max_rel_diff"] for r in results))
    S["guards"] = g
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("struct", choices=("S1", "S2"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="sid:planet, for a quick trial")
    a = ap.parse_args()
    os.environ["PLANETSENSE_STRUCTURE"] = a.struct
    tks = tasks(a.struct)
    if a.only:
        s_, p_ = a.only.split(":")
        tks = [t for t in tks if t["sid"] == s_ and t["planet"] == p_]
    print(len(tks), "walking cases on", a.struct, flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = []
        for r in pool.imap(run_task, tks):
            results.append(r)
            print(f"{r['sid']:16s} {r['planet']:5s} {r['wall']:.0f} s | reimpl diff {r['diag']['reimpl_max_rel_diff']:.1e}", flush=True)
    out = dict(status="SIMULATION - post hoc checks on existing records (Phase M)", structure=a.struct,
               summary=summarize(results), data=results, wall_s=time.time() - t0)
    name = f"walking_checks_{a.struct}" + ("_trial" if a.only else "") + ".json"
    json.dump(out, open(T(name), "w"), indent=1, default=float)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
