"""Phase D (standing, calibrated body): how gravity changes what a standing
person's wearable sensors receive from the floor, with the body's vertical
dynamics calibrated to published apparent-mass summary values at Earth gravity.

STATUS: simulation.  Earth values are calibrated to published summary
statistics (not validated against an individual); Mars and Moon values are
model-based extrapolations whose spread over the listed variants is the
honest statement of what the model does and does not pin down.

Variants (each at Earth, Mars, Moon)
  nominal        leg exponent 1.5, constant leg damping, impedance scaled with local weight
  earth_fixed    as nominal, but Earth joint impedance kept on every planet
  leg_n1 / n2    leg stiffening exponent 1 (linear) / 2, re-calibrated to the same Earth targets
  soft_damped    calibrated to 5.24 Hz, 39 %   (published equivalent single-degree-of-freedom values)
  stiff_light    calibrated to 6.6 Hz, 22 %
  prop_damping   leg damping proportional to the leg force
  fully_linear   linear leg + linear foot pad + Earth impedance (control: nothing in the body
                 can follow the weight, so any remaining gravity effect is geometric)
  rigid_leg      first prototype without leg compliance (reference for what the calibration changed)

    python experiments/phase_D_standing/standing_v2.py
"""
import json
import os
import subprocess
import time

import numpy as np

from planetsense.contact.compliant import ContactParams, tangent_stiffness
from planetsense.evaluation.linear import (apparent_mass, base_transmissibility, com_participation,
                                     fit_sdof, frf, heavy_platform)
from planetsense.human.body_v2 import load_leg
from planetsense.sensor.imu import ug_to_si
from planetsense.simulation.standing import make_standing
from planetsense.structure.spine_beam import SpineBeamDeck

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}
nominal_pad = ContactParams()
linear_pad = ContactParams(k_n=tangent_stiffness(nominal_pad, 75.0 * 9.81 / 8), n_exp=1.0,
                           c_n=nominal_pad.c_n, k_t=nominal_pad.k_t, c_t=nominal_pad.c_t, mu=nominal_pad.mu)
VARIANTS = {
    "nominal": dict(leg="nominal_n1.5_constant"),
    "earth_fixed": dict(leg="nominal_n1.5_constant", gain_policy="earth_fixed"),
    "leg_n1": dict(leg="nominal_n1_constant"),
    "leg_n2": dict(leg="nominal_n2_constant"),
    "soft_damped": dict(leg="soft_damped_n1.5_constant"),
    "stiff_light": dict(leg="stiff_light_n1.5_constant"),
    "prop_damping": dict(leg="nominal_n1.5_proportional"),
    "fully_linear": dict(leg="nominal_n1_constant", gain_policy="earth_fixed", contact=linear_pad),
    "rigid_leg": dict(leg=None),
}
SENS = ("pelvis_L", "wrist_L", "thigh_L")


X_STAND = 4.2        # m along the 14 m span: no vertical bending mode below 30 Hz has a node here


def build(v, g, deck, x_com=7.0):
    kw = dict(VARIANTS[v])
    leg = kw.pop("leg")
    return make_standing(g=g, deck=deck, leg=load_leg(leg) if leg else None, x_com=x_com, **kw)


out = dict(status="SIMULATION - Earth calibrated to published summary values, Mars/Moon extrapolated - not validated",
           started=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           commit_at_run=subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           variants={k: {kk: (vv.asdict() if hasattr(vv, "asdict") else vv) for kk, vv in v.items()} for k, v in VARIANTS.items()},
           gravities=G, standing_position_m=X_STAND)
curves = {}
fb = np.logspace(np.log10(0.5), np.log10(40.0), 300)
fam = np.linspace(0.5, 20.0, 100)
curves["f_base"], curves["f_am"] = fb, fam
plat = heavy_platform()
fe = SpineBeamDeck()
deck = fe.modal(f_max=150.0)
keep = [i for i, f in enumerate(deck.freq_hz) if f < 40.0]
out["deck_modes"] = [dict(label=deck.labels[i], f=float(deck.freq_hz[i]), zeta=float(deck.zeta[i])) for i in keep]
fl = np.arange(0.5, 40.0001, 0.05)
curves["f_e1"] = fl
S_noise = ug_to_si(100.0) ** 2
out["sensor_noise_density_ug"] = 100.0
rows = []
for v in VARIANTS:
    for gname, g in G.items():
        t0 = time.perf_counter()
        # ---- body alone on a heavy platform
        sc = build(v, g, plat)
        nq = sc.system.nq
        T, names, A = base_transmissibility(sc, fb)
        part = com_participation(sc.human, sc.y_eq[:nq], A)
        bounce = max((m for m in part if m["f"] > 1.5), key=lambda m: m["rz"])
        M = apparent_mass(sc, fam) / sc.human.total_mass
        f_am, z_am = fit_sdof(fam, M)
        r = dict(variant=v, gravity=gname, g=g, stable=bool(np.linalg.eigvals(A).real.max() < 1e-6),
                 bounce_f=bounce["f"], bounce_zeta=bounce["zeta"], bounce_rz=bounce["rz"],
                 apparent_mass_f=f_am, apparent_mass_zeta=z_am,
                 apparent_mass_peak=float(np.abs(M).max()), apparent_mass_peak_f=float(fam[np.abs(M).argmax()]),
                 leg_compression_mm=float(1e3 * sc.info["leg_compression"].mean()) if sc.info["leg_compression"].size else None,
                 pad_compression_mm=float(1e3 * sc.info["penetration"].mean()))
        curves[f"AM|{v}|{gname}"] = np.abs(M)
        for sname in SENS:
            i = names.index(sname)
            tz = np.abs(T[:, i, 2, 2])
            curves[f"T|{v}|{gname}|{sname}_zz"] = tz
            r[f"{sname}_zz_peak"] = float(tz.max())
            r[f"{sname}_zz_peak_f"] = float(fb[tz.argmax()])
        curves[f"T|{v}|{gname}|pelvis_L_xx"] = np.abs(T[:, names.index("pelvis_L"), 0, 0])
        # ---- on the footbridge deck
        sc = build(v, g, deck, X_STAND)
        s, y = sc.system, sc.y_eq
        A, BF, _ = s.linearize(y)
        lam = np.linalg.eigvals(A)
        lam = lam[lam.imag > 0]
        modes = []
        C, D, _, names = s.sensor_jacobians(y)
        nm = s.nm
        rr = slice(2 * nq + nm, 2 * nq + 2 * nm)
        Sf = s.S2.reshape(s.npts, 3, nm).mean(axis=0)
        H = frf(A, BF, np.vstack([C, Sf @ A[rr]]), np.vstack([D, Sf @ BF[rr]]), fl)
        P = np.sum(np.abs(H) ** 2, axis=2)                 # response PSD per unit PSD of nine independent forces
        ns = len(names)
        for sname in SENS:
            curves[f"E1|{v}|{gname}|{sname}_z"] = P[:, 6 * names.index(sname) + 2]
        curves[f"E1|{v}|{gname}|floor_z"] = P[:, 6 * ns + 2]
        for i in keep:
            f0, z0 = deck.freq_hz[i], deck.zeta[i]
            w0 = 2 * np.pi * f0
            j = np.argmin(np.abs(lam - (-z0 * w0 + 1j * w0 * np.sqrt(1 - z0 ** 2))))
            band = np.abs(fl - f0) <= max(0.1, 0.03 * f0)
            tot = lambda a: float(np.trapezoid(a[band], fl[band]))
            md = dict(mode=deck.labels[i], f_bare=float(f0), zeta_bare=float(z0),
                      f_occupied=float(abs(lam[j]) / (2 * np.pi)), zeta_occupied=float(-lam[j].real / abs(lam[j])),
                      floor_z=tot(P[:, 6 * ns + 2]))
            for sname in SENS:
                k = names.index(sname)
                md[f"{sname}_z_over_floor_z"] = float(np.sqrt(tot(P[:, 6 * k + 2]) / max(md["floor_z"], 1e-300)))
                md[f"{sname}_3axis_over_floor_z"] = float(np.sqrt(tot(P[:, 6 * k:6 * k + 3].sum(axis=1)) / max(md["floor_z"], 1e-300)))
                # shaker force PSD (N^2/Hz per component) at which the sensor's band-mean PSD equals its noise PSD
                md[f"{sname}_force_psd_unit_snr"] = float(S_noise / np.mean(P[band, 6 * k:6 * k + 3].sum(axis=1)))
            modes.append(md)
        r["deck_modes"] = modes
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
        print(f"{v:13s} {gname:5s} bounce {bounce['f']:.2f} Hz z={bounce['zeta']:.2f} | AM fit {f_am:.2f} Hz {z_am:.2f} | "
              f"pelvis zz peak {r['pelvis_L_zz_peak']:.2f} @ {r['pelvis_L_zz_peak_f']:.1f} Hz | {r['wall_s']} s", flush=True)
out["rows"] = rows
out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
os.makedirs(os.path.join(ROOT, "outputs", "tables"), exist_ok=True)
with open(os.path.join(ROOT, "outputs", "tables", "standing_v2_summary.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
np.savez_compressed(os.path.join(ROOT, "outputs", "processed", "standing_v2_curves.npz"), **curves)
print("wrote outputs/tables/standing_v2_summary.json")
