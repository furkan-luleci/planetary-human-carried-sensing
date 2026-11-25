"""Build the offline results viewer: ONE self-contained HTML file.

    python viewer/build_viewer.py

Reads the simulation outputs (records, tables) and inlines them, together with
the page's own script and style, into viewer/Planetary_Sensing_Viewer.html.  The
page loads nothing from the network.  Everything it shows is pre-computed; the
page never simulates.

Two structures: the habitat floor (the study's main case; "S2" in the code and result
files, records in outputs/raw/records_S2) and the reference bridge, the earlier study's
footbridge ("S1", outputs/raw/records).  The keys never reach the page.  The floor's walks
are paired with the deck's rigid-ground twins, cut to the floor's crossing.  A habitat-floor
result file that does not exist yet is skipped.
"""
import base64
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.signal import decimate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from planetsense.identification.spectral import ensemble_csd, sv1          # noqa: E402
from planetsense.simulation.records import DECK_L, analysis_window, load_record    # noqa: E402

# S1 (the reference bridge, the earlier study's footbridge) and S2 (the habitat floor, the main case) are internal
# keys only; the page shows "Reference bridge (the earlier study's footbridge)" and "Habitat floor (NASA Common Habitat)".
REC = os.environ.get("PLANETSENSE_REC_DIR", os.path.join(ROOT, "outputs", "raw", "records"))
REC_S2 = os.environ.get("PLANETSENSE_REC_S2_DIR", os.path.join(ROOT, "outputs", "raw", "records_S2"))
TAB = os.path.join(ROOT, "outputs", "tables")
SRC = os.path.join(ROOT, "viewer", "src")
PLANETS = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}
SENSORS = {"wrist": "wrist_L", "thigh": "thigh_L"}
FMAX = 40.0


def enc(a):
    """Quantise to 16-bit integers with one scale factor (decoded in the page)."""
    a = np.asarray(a, np.float64)
    m = float(np.abs(a).max()) if a.size else 0.0
    s = (m / 32000.0) if m > 0 else 1.0
    q = np.round(a / s).astype("<i2")
    return dict(b=base64.b64encode(q.tobytes()).decode(), s=s, shape=list(a.shape))


def f32(a):
    return dict(f=base64.b64encode(np.asarray(a, "<f4").tobytes()).decode(), shape=list(np.shape(a)))


def rnd(x, n=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(f"{x:.{n}g}")
    return x


def spectra(rec, win=None):
    """First-singular-value spectra of each sensor's three accelerometer axes
    and the vertical deck acceleration under the person, (m/s^2)^2/Hz."""
    fs = float(rec["fs"])
    n = len(rec["t"])
    m = np.ones(n, bool) if win is None else win
    nper = int((16.0 if rec["activity"] == "standing" else 4.0) * fs)
    out = {}
    names = list(rec["sensors"])
    f = None
    for key, sn in SENSORS.items():
        X = rec["f"][m, names.index(sn), :].astype(float)
        f, G, _ = ensemble_csd([X], fs, nper)
        out[key] = sv1(G)
    if "a_under" in rec:
        f, G, _ = ensemble_csd([rec["a_under"][m][:, 2:3].astype(float)], fs, nper)
        out["deck"] = G[:, 0, 0].real
    k = f <= FMAX
    return f[k], {a: b[k] for a, b in out.items()}


def window(rec):
    """Analysis window of the study (walking: the crossing minus its first 1.5 s;
    standing: the 60 s after the lead-in)."""
    return analysis_window(rec)


def trial(rec, twin=None, structure="S1"):
    planet = str(rec["planet"])
    g = PLANETS[planet]
    W = float(rec["mass"]) * g
    act = str(rec["activity"])
    ta = rec["t_anim"]
    kp = rec["kp"].astype(float)                       # (n, nk, 3)
    L = float(rec.get("deck_L", DECK_L))               # span of the trial's structure (rigid twins: the deck's)
    if act == "walking":
        x = kp[:, 3, 0]
        keep = (x > -2.0) & (x < L + 1.3)              # from 2 m before the structure to 1.3 m past its far end
        t_view = (ta[keep][0], ta[keep][-1])
    else:
        t_s = float(rec.get("t_analysis_start", 4.0) or 4.0)          # after the coupled lead-in
        t_view = (t_s, t_s + 20.0)
        keep = (ta >= t_view[0]) & (ta <= t_view[1])
    ta, kp = ta[keep], kp[keep]
    refx = kp[:, 3, 0].copy() if act == "walking" else np.full(len(ta), kp[:, 3, 0].mean())
    relx = kp[:, :, 0] - refx[:, None]
    z = kp[:, :, 2]
    yy = kp[:, :, 1]
    x0, z0, y0 = relx.mean(axis=0), z.mean(axis=0), yy.mean(axis=0)
    tr = dict(id=structure + ":" + str(rec["name"]), structure=structure, L=rnd(L, 6), planet=planet, activity=act,
              exc=str(rec["excitation"]), model3d=str(rec.get("walker", "")) == "3d",
              cond=str(rec.get("condition", "")), g=g, weight=rnd(W), t0=float(ta[0]), fsAnim=50,
              nAnim=int(len(ta)), kpNames=[str(s) for s in rec["kp_names"]],
              refx=f32(refx), x0=rnd(x0, 6), z0=rnd(z0, 6), y0=rnd(y0, 6), dx=enc(relx - x0), dz=enc(z - z0), dy=enc(yy - y0),
              fz=enc(rec["fc_anim"][keep][:, :, 2] / W), fx=enc(rec["fc_anim"][keep][:, :, 0] / W))
    if "deck_w" in rec:
        tr["deckX"] = rnd(rec["deck_x"], 5)
        tr["deckW"] = enc(rec["deck_w"][keep])
    if act == "walking" and rec.get("x_start") is not None:
        tr["xStart"] = rnd(float(rec["x_start"]), 4)          # where the walk starts (rigid ground before the structure)
    if act == "standing" and rec.get("x_com") is not None:
        tr["standX"] = rnd(float(rec["x_com"]), 4)
    if rec.get("lead_s") is not None:
        tr["leadS"] = rnd(float(rec["lead_s"]), 4)             # lead-in before the analysed part (s)
    # signals at 100 Hz for the time-series plots
    t = rec["t"]
    ks = (t >= t_view[0]) & (t <= t_view[1])
    names = list(rec["sensors"])
    sig = {}
    for key, sn in SENSORS.items():
        a = rec["f"][ks, names.index(sn), :].astype(float)
        mag = np.linalg.norm(a, axis=1)
        sig[key] = enc(decimate(mag - mag.mean(), 2, ftype="fir", zero_phase=True))
    if "a_under" in rec:
        sig["deck"] = enc(decimate(rec["a_under"][ks, 2].astype(float), 2, ftype="fir", zero_phase=True))
    sig["grfL"] = enc(decimate(rec["fc"][ks, 0, 2].astype(float) / W, 2, ftype="fir", zero_phase=True))
    sig["grfR"] = enc(decimate(rec["fc"][ks, 1, 2].astype(float) / W, 2, ftype="fir", zero_phase=True))
    if str(rec.get("walker", "")) == "3d":       # sideways ground force (both feet), 3-D walker only
        sig["grfY"] = enc(decimate(rec["fc"][ks, :, 1].astype(float).sum(axis=1) / W, 2, ftype="fir", zero_phase=True))
    tr["sig"] = dict(fs=100, t0=float(t[ks][0]), ch=sig)
    # spectra
    w = window(rec)
    f, sp = spectra(rec, w)
    tr["spec"] = dict(f=rnd(f, 6), ch={k: f32(v) for k, v in sp.items()})
    if twin is not None:
        f2, sp2 = spectra(twin, window(twin))
        tr["spec"]["rigid"] = {k: f32(np.interp(f, f2, v)) for k, v in sp2.items() if k != "deck"}
    st = dict(energy_drift_rel=rnd(float(rec["energy_drift_rel"])), fell=bool(rec["fell"]))
    if act == "walking":
        gm = rec["gait"]
        for k in ("speed", "step_frequency", "step_length", "duty_factor", "double_support_fraction",
                  "flight_fraction", "grf_foot_peak_over_W", "grf_peak_over_W", "com_vertical_excursion", "gait",
                  "swing_scuffs_per_step"):
            st[k] = rnd(gm.get(k))
        st["dlf"] = rnd(gm.get("dlf"))
        for k in ("step_width", "grf_lateral_peak_over_W", "dlf_lat", "com_lateral_excursion", "heading_max", "trunk_lean"):
            if gm.get(k) is not None:
                st[k] = rnd(gm.get(k))
        ev = rec["events"]
        tr["steps"] = rnd([[e[0], int(e[1]), e[2]] for e in ev if t_view[0] <= e[0] <= t_view[1]], 5)
    else:
        st.update({k: rnd(v) for k, v in rec["contact"].items()})
    if "a_under" in rec:
        st["deck_rms_z"] = rnd(float(rec["a_under"][w, 2].std()))
    for key, sn in SENSORS.items():
        a = rec["f"][w, names.index(sn), :].astype(float)
        st[f"{key}_rms"] = rnd(float(np.sqrt(((a - a.mean(axis=0)) ** 2).sum(axis=1).mean())))
    tr["stats"] = st
    return tr


def font_faces():
    """The viewer's lettering: B612 and B612 Mono, the typefaces designed for aircraft cockpit displays
    (SIL Open Font License 1.1, viewer/fonts/OFL-B612*.txt), embedded so that the page looks the same
    offline on every computer."""
    import base64
    out = ["/* B612 and B612 Mono, Copyright 2012 The B612 Project Authors (https://github.com/polarsys/b612), "
           "SIL Open Font License 1.1 (viewer/fonts/OFL-B612.txt) */"]
    fdir = os.path.join(ROOT, "viewer", "fonts")
    for fam, files in (("B612", (("Regular", 400, "normal"), ("Italic", 400, "italic"), ("Bold", 700, "normal"))),
                       ("B612 Mono", (("Regular", 400, "normal"), ("Bold", 700, "normal")))):
        for style, weight, ital in files:
            path = os.path.join(fdir, f"{fam.replace(' ', '')}-{style}.ttf")
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            out.append(f"@font-face{{font-family:'{fam}';src:url(data:font/ttf;base64,{b64}) format('truetype');"
                       f"font-weight:{weight};font-style:{ital};font-display:block}}")
    return "\n".join(out) + "\n"


PLAIN = (("Model of the structure (S1 footbridge)", "Model of the reference bridge (the earlier study\u2019s bridge model)"),
         ("Second structure (habitat floor): S1 unchanged by the generalised code; S2's settings follow the declared rules",
          "Habitat floor: the reference bridge is unchanged by the generalised code, and the floor\u2019s settings follow the declared rules"),
         ("S1 footbridge", "reference bridge (the earlier study\u2019s bridge model)"), ("S2L", "lunar-weight floor"),
         ("S2's", "the habitat floor\u2019s"), ("S1's", "the reference bridge\u2019s"),
         ("Earth gait slowed down, shakers on", "What-if: Earth 1.25 m/s gait scaled, shakers on"),
         ("Earth gait slowed down, footsteps only", "What-if: Earth 1.25 m/s gait scaled, footsteps only"),
         ("What-if case: the Earth gait replayed in slow motion (equal Froude number).",
          "What-if case: the Earth gait tuned for 1.25 m/s, replayed in slow motion (equal Froude number); on Earth, the 1.25 m/s gait itself."))


def plain(x):
    """Reader-facing wording for text taken from the result files: the internal keys S1 / S2 / S2L never reach the page."""
    import re
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [plain(v) for v in x]
    if not isinstance(x, str):
        return x
    for a, b in PLAIN:
        x = x.replace(a, b)
    x = re.sub(r"\bS1\b", "the reference bridge", x)
    return re.sub(r"\bS2\b", "the habitat floor", x)


SCRUB = (("the earlier study", "the earlier study"), ("earlier-study baseline", "Earlier study's baseline"),
         ("Earth gait slowed down", "What-if: Earth 1.25 m/s gait scaled"))


def scrub(x):
    """Text-only replacements on every string value of the bundle (keys and ids untouched)."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str):
        for a, b in SCRUB:
            x = x.replace(a, b)
    return x


def finite(x):
    """JSON for the browser: a non-finite number (NaN in a result file) becomes null."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def structure_info(name):
    """Geometry and settings of a structure for the drawings and texts, read from the model code
    (planetsense.structure.registry), so that no number is typed into the page."""
    from planetsense.excitation.multisine import shaker_layout
    from planetsense.structure import registry as SR
    md, st = SR.modal_deck(name), SR.get(name)
    L, W = float(md.params.L), float(md.params.W)
    out = dict(L=rnd(L, 6), W=rnd(W, 5), line_y=rnd(float(st.line_y), 4), stand=rnd([float(v) for v in st.stand_xy], 5),
               shakers=rnd(shaker_layout(L, W).tolist(), 5), shaker_rms=rnd(float(st.shaker_rms), 4),
               floor_rms=rnd(float(st.floor_rms), 4), n_modes=int(len(md.freq_hz)), zeta1=rnd(float(md.zeta[0]), 4),
               modes=[dict(label=str(l), f=rnd(float(f), 6), zeta=rnd(float(z), 4))
                      for l, f, z in zip(md.labels, md.freq_hz, md.zeta) if f < FMAX])
    design = md.meta.get("design")
    if design:                                                   # the habitat floor (habitat_deck.HabitatDeck)
        from planetsense.structure.habitat_deck import HabitatDeck
        hd = HabitatDeck()
        n = int(design["beam"]["n_spans"])
        R = 0.5 * float(hd.D_vessel)
        out.update(supports=rnd([L * i / n for i in range(n + 1)], 6), span=rnd(float(design["beam"]["L"]), 5),
                   deck_t=rnd(float(hd.t_deck), 4), m_area=rnd(float(design["m_area"]), 4), mass_strip=rnd(float(design["total_mass_strip"]), 4),
                   beam_spacing=rnd(float(hd.s_beam), 4),
                   # the floor spans the vessel's chord: its height below the vessel axis follows from the span and the diameter
                   vessel=dict(D=rnd(float(hd.D_vessel), 4), z_axis=rnd(float(np.sqrt(R * R - 0.25 * L * L)), 5), hatch=rnd(float(hd.hatch), 4)))
    else:                                                        # the reference bridge (spine_beam.SpineBeamDeck)
        p = md.params
        out.update(supports=[0.0, rnd(L, 6)], span=rnd(L, 6), deck_t=rnd(float(p.t), 4), m_area=rnd(float(p.rho * p.t), 4))
    return out


def mode_shapes(name):
    """Mode shapes of a structure's surface on a 29 x 3 grid, for the explanatory animation."""
    from planetsense.structure import registry as SR
    dk = SR.modal_deck(name)
    gx = np.linspace(0.0, dk.params.L, 29)
    gy = np.array([-0.5 * dk.params.W, 0.0, 0.5 * dk.params.W])
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    Sg = dk.surface_modes(X.ravel(), Y.ravel())                    # (87, 3, nm)
    shapes = []
    for i, f in enumerate(dk.freq_hz):
        if f >= FMAX:
            continue
        u = Sg[:, :, i]
        u = u / np.abs(u).max()
        shapes.append(dict(label=str(dk.labels[i]), f=rnd(float(f), 5), y=rnd(u[:, 1], 3), z=rnd(u[:, 2], 3)))
    return dict(nx=29, ny=3, L=float(dk.params.L), W=float(dk.params.W), modes=shapes)


def main():
    recs = {}
    for p in sorted(glob.glob(os.path.join(REC, "*.npz"))):
        r = load_record(p)
        recs[str(r["name"])] = r
    trials = []
    # the habitat floor (main case): repetition 0 of every case; its walks pair with the reference bridge's rigid-ground
    # twins, cut to the floor's crossing (as experiments/phase_K_S2/step_timing_s2.py does)
    for p in sorted(glob.glob(os.path.join(REC_S2, "*_r0.npz"))):
        r = load_record(p)
        if int(r["rep"]) != 0:
            continue
        twin = None
        if str(r["activity"]) == "walking" and str(r["excitation"]) != "rigid":
            tw = recs.get(f"walking_{r['condition']}_rigid_r0")
            twin = dict(tw, deck_L=float(r["deck_L"])) if tw is not None else None
        trials.append(trial(r, twin, "S2"))
    n_floor = len(trials)
    for name, r in recs.items():
        # the earlier-study-baseline check cases (mid-span standing, 1.5 m/s walking) appear in the results pages only
        if int(r["rep"]) != 0 or "Earth-mid" in name or str(r.get("condition", "")) == "earth_1.50":
            continue
        twin = None
        if str(r["activity"]) == "walking" and str(r["excitation"]) != "rigid":
            twin = recs.get(name.replace("_E1_", "_rigid_").replace("_E2_", "_rigid_"))
        trials.append(trial(r, twin, "S1"))
    # every simulated trial of the study (reference bridge, habitat floor and the floor sized for lunar weight)
    n_all = sum(len(glob.glob(os.path.join(ROOT, "outputs", "raw", d, "*.npz"))) for d in ("records", "records_S2", "records_S2L"))
    cnt = os.path.join(ROOT, "outputs", "raw", "record_count.json")   # the full count, when only repetition 0 is stored
    if os.path.exists(cnt):
        n_all = max(n_all, json.load(open(cnt))["n_records"])
    bundle = dict(
        meta=dict(built=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                  commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
                  n_records=n_all),
        planets={k: dict(g=v) for k, v in PLANETS.items()},
        trials=trials)
    bundle["structs"] = {n: structure_info(n) for n in ("S2", "S1")}
    # mode shapes of both structures for the explanatory animation (surface motion on a 29 x 3 grid)
    bundle["shapes"] = {n: mode_shapes(n) for n in ("S2", "S1")}
    # standing study (linear analysis)
    p = os.path.join(TAB, "standing_v2_summary.json")
    if os.path.exists(p):
        S = json.load(open(p))
        C = np.load(os.path.join(ROOT, "outputs", "processed", "standing_v2_curves.npz"))
        cur = {}
        for k in C.files:
            if k.startswith("T|") or k.startswith("AM|"):
                cur[k] = f32(C[k])
        bundle["standing"] = dict(f_base=rnd(C["f_base"], 5), f_am=rnd(C["f_am"], 5), curves=cur,
                                  rows=rnd(S["rows"]), deck_modes=S["deck_modes"], status=S["status"],
                                  position=S.get("standing_position_m"))
    # gait tuning results
    gait = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "gait", "*.json"))):
        if os.path.basename(p).startswith("_") or ".ckpt" in p:
            continue
        j = json.load(open(p))
        gait[os.path.basename(p)[:-5]] = dict(g=j["g"], target_speed=j["target_speed"], targets=j.get("targets"),
                                              metrics=rnd(j.get("metrics_validation_30s")), cost=rnd(j.get("cost")),
                                              kind=j.get("kind", "tuned"), params=rnd(j.get("params")))
    bundle["gait"] = gait
    for key, fn in (("outcomes", "outcomes.json"), ("verification", "verification_v2.json"),
                    ("profiles", "gait_profiles.json"), ("mounting", "mounting_check.json"),
                    ("legacy", "legacy_oi_baseline.json"), ("threshold_check", "threshold_check.json")):
        p = os.path.join(TAB, fn)
        if os.path.exists(p):
            bundle[key] = json.load(open(p))
    for key, fn in (("step_timing", "step_timing.json"), ("step_timing_check", "step_timing_check.json"),
                    ("step_timing_jitter", "step_timing_jitter.json"), ("step_timing_clocks", "step_timing_clocks.json")):
        p = os.path.join(TAB, fn)
        if os.path.exists(p):
            bundle[key] = json.load(open(p))
    if "step_timing" in bundle:                    # keep what the page draws
        for byp in bundle["step_timing"]["data"].values():
            for r in byp.values():
                for var in ("exact", "device"):
                    for d in r[var].values():
                        d.pop("S", None); d.pop("B", None)
                r["timing"] = {k: dict(n_e=v["n_e"], stride_s=v["stride_s"]) for k, v in r.get("timing", {}).items()}
    if "step_timing_check" in bundle:
        for k in ("cells", "m2_cells"):
            bundle["step_timing_check"].pop(k, None)
        bundle["step_timing_check"]["criterion1"].pop("changed", None)
    if "step_timing_clocks" in bundle:
        bundle["step_timing_clocks"] = dict(summary=bundle["step_timing_clocks"]["summary"])
    if "legacy" in bundle:                         # the display spectra are not drawn in the page
        for byp in bundle["legacy"]["data"].values():
            for r in byp.values():
                r.pop("spectra", None)
    # ---- the habitat floor (main case): its own result files; a file that does not exist (yet) is skipped
    def load_tab(fn):
        path = os.path.join(TAB, fn)
        return json.load(open(path)) if os.path.exists(path) else None
    o2 = load_tab("outcomes_S2.json")
    if o2:
        for byp in o2["data"].values():
            for d in byp.values():
                for sk in ("wrist", "thigh"):
                    for k in ("rec", "rec_alt"):
                        d.get(sk, {}).get(k, {}).pop("f1_per_record", None)
        bundle["outcomes_S2"] = o2
    mn2 = load_tab("mounting_check_S2.json")
    if mn2:
        bundle["mounting_S2"] = mn2
    st2 = load_tab("step_timing_S2.json")
    if st2:                                        # keep what the page draws
        for byp in st2["data"].values():
            for r in byp.values():
                for v in r["m2"].values():
                    for d in v.values():
                        d.pop("rejection", None)
                        for k in ("f", "S", "B"):
                            d[k] = rnd(d[k], 4)
                r["timing"] = {k: dict(n_e=v["n_e"], stride_s=rnd(v["stride_s"]), random_ms_median=rnd(v["random_ms_median"]))
                               for k, v in r.get("timing", {}).items()}
        bundle["step_timing_S2"] = st2
    ck2 = load_tab("step_timing_clocks_S2.json")
    if ck2:
        bundle["step_timing_clocks_S2"] = dict(summary=ck2["summary"])
    jx2 = load_tab("step_timing_jitter_S2.json")   # written after the main results (post hoc); skipped while absent
    if jx2:
        for r in jx2.get("results", []):
            r.pop("wall", None)
        bundle["step_timing_jitter_S2"] = jx2
    oc = load_tab("occupied_curves.json")
    if oc:
        bundle["occupied_curves"] = dict(
            linear={n: {k: (f32(v) if isinstance(v, list) else v) for k, v in d.items()} for n, d in oc["linear"].items()},
            trials={pl: dict(f=rnd(v["f"], 6), psd=f32(v["psd"]), n_records=v["n_records"], peak_hz=v["peak_hz"])
                    for pl, v in oc["trials"].items()})
    for key, fn in (("hsi", "hsi_structures.json"), ("phaseK", "phaseK_check.json")):
        j = load_tab(fn)
        if j:
            bundle[key] = j
    # text that reaches the reader from the result files: no internal structure keys
    for key in ("verification", "mounting", "mounting_S2"):
        if key in bundle:
            bundle[key] = plain(bundle[key])
    for key in ("outcomes", "outcomes_S2"):
        if key in bundle:
            bundle[key]["scenarios"] = plain(bundle[key]["scenarios"])
    tst = os.path.join(ROOT, "outputs", "tables", "structures.csv")
    if os.path.exists(tst):                                    # the habitat floor against the reference bridge
        import csv as _csv
        rows = plain(list(_csv.reader(open(tst))))
        bundle["structures"] = dict(header=rows[0], rows=rows[1:])
    hab = os.path.join(ROOT, "viewer", "assets", "setting.png")   # the setting, redrawn for the viewer by viewer/make_viewer_figures.py
    if os.path.exists(hab):
        bundle["habitat_png"] = base64.b64encode(open(hab, "rb").read()).decode()
    html = open(os.path.join(SRC, "index.html")).read()
    html = html.replace("/*__STYLE__*/", font_faces() + open(os.path.join(SRC, "style.css")).read())
    parts = ("charts.js", "scene.js", "scene3d.js", "app.js", "results.js")
    node = subprocess.run(["which", "node"], capture_output=True, text=True).stdout.strip()
    if node:                                   # refuse to bundle a script that does not parse
        for f in parts:
            chk = subprocess.run([node, "--check", os.path.join(SRC, f)], capture_output=True, text=True)
            if chk.returncode != 0:
                raise SystemExit(f"viewer/src/{f} does not parse:\n{chk.stderr}")
    else:
        print("warning: node not found; the viewer's scripts were not syntax-checked")
    js = "\n".join(open(os.path.join(SRC, f)).read() for f in parts)
    html = html.replace("/*__SCRIPT__*/", js)
    three = open(os.path.join(ROOT, "viewer", "vendor", "three.min.js")).read()
    html = html.replace("/*__THREE__*/", three.replace("</script", "<\\/script"))
    bundle = scrub(bundle)                          # internal labels in bundled result files, shown or not
    html = html.replace("/*__BUNDLE__*/", json.dumps(finite(bundle), separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    out = os.environ.get("PLANETSENSE_VIEWER_OUT", os.path.join(ROOT, "viewer", "Planetary_Sensing_Viewer.html"))
    open(out, "w").write(html)
    print("wrote", os.path.relpath(out, ROOT), "%.2f MB" % (os.path.getsize(out) / 1e6), "| trials:", len(trials),
          f"({n_floor} habitat floor, {len(trials) - n_floor} reference bridge)",
          "| habitat-floor tables:", ", ".join(k for k in ("outcomes_S2", "mounting_S2", "step_timing_S2", "step_timing_clocks_S2",
                                                          "step_timing_jitter_S2", "occupied_curves", "hsi", "phaseK") if k in bundle))


if __name__ == "__main__":
    main()
