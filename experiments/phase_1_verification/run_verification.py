"""Phase 1 -- numerical verification of the articulated-human / flexible-deck
prototype (gates G1-G3, standing only).

Writes measured values (not just pass/fail) to
    outputs/tables/verification_summary.json
    outputs/processed/verification_curves.npz

Nothing here is a planetary finding: Moon and Mars values appear only to check
that the equations respond to gravity as they must.

    python experiments/phase_1_verification/run_verification.py
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from planetsense.contact.compliant import ContactParams                      # noqa: E402
from planetsense.coupling.system import CoupledSystem, rk4_max_stable_dt     # noqa: E402
from planetsense.evaluation.linear import com_participation, frf           # noqa: E402
from planetsense.excitation.multisine import make_multisine                  # noqa: E402
from planetsense.human.control import PostureController, standing_posture    # noqa: E402
from planetsense.human.model import build_human                              # noqa: E402
from planetsense.simulation.standing import make_standing                    # noqa: E402
from planetsense.structure.spine_beam import SpineBeamDeck                   # noqa: E402
from tolerances import TOL                                             # noqa: E402

G = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}
out = dict(tolerances=TOL, started=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
curves = {}


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


import scipy, numba, mujoco                                            # noqa: E402,E401
_dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "src", "tests", "experiments"],
                        capture_output=True, text=True).stdout.strip()
out["provenance"] = dict(commit_at_run=git_head(), worktree_dirty=bool(_dirty),
                         python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__,
                         pinocchio=pin.__version__, numba=numba.__version__,
                         mujoco=mujoco.__version__, cpu_count=os.cpu_count())

fe = SpineBeamDeck()
deck = fe.modal(f_max=150.0)
out["deck"] = dict(n_modes=deck.nm, f_cut=150.0, alphaM=fe.alphaM, betaK=fe.betaK,
                   freqs_below_50=[float(f) for f in deck.freq_hz if f < 50],
                   labels_below_50=[l for l, f in zip(deck.labels, deck.freq_hz) if f < 50])

# ------------------------------------------------------------- V1 static support
rows = []
for name, g in G.items():
    for floor in ("rigid", "S1 deck"):
        sc = make_standing(g=g, deck=deck if floor == "S1 deck" else None)
        s, y, info = sc.system, sc.y_eq, sc.info
        h = sc.human
        dy = s.rhs(0.0, y, first=True)
        A, _, _ = s.linearize(y)
        lam = np.linalg.eigvals(A)
        part = com_participation(h, y[:36], A)
        sway = max((m for m in part if m["f"] < 1.0), key=lambda m: m["rx"])
        bounce = max((m for m in part if m["f"] > 2.0), key=lambda m: m["rz"])
        k1 = s.rhs(0.0, y, first=True)
        from planetsense.sensor.imu import imu_ideal
        imu = imu_ideal(h, y[:36], y[36:72], k1[36:72])
        rows.append(dict(
            gravity=name, g=g, floor=floor, weight_N=info["weight"],
            sumFn_minus_W_rel=float(abs(info["sum_fn"] - info["weight"]) / info["weight"]),
            max_state_rate_at_eq=float(np.abs(dy[:s.nx]).max()),
            root_residual_max=float(np.abs(info["residual_root"]).max()),
            penetration_mm=float(1e3 * info["penetration"].mean()),
            deck_deflection_mm=float(-1e3 * (s.S2 @ info["eta"]).reshape(-1, 3)[:, 2].mean()) if s.nm else 0.0,
            cop_minus_com_m=float(np.abs(info["cop"] - info["com"][:2]).max()),
            tau_ff_max_Nm=float(np.abs(info["tau_ff"]).max()),
            ankle_tau_ff_Nm=float(info["tau_ff"][h.dof("ankle_L_ry") - 6]),
            max_real_eig=float(lam.real.max()),
            rk4_max_stable_dt_ms=float(1e3 * rk4_max_stable_dt(A)),
            sway_mode=dict(f=sway["f"], zeta=sway["zeta"], rx=sway["rx"]),
            bounce_mode=dict(f=bounce["f"], zeta=bounce["zeta"], rz=bounce["rz"]),
            specific_force_norm_minus_g=float(max(abs(np.linalg.norm(v["f"]) - g) for v in imu.values()))))
out["static_support"] = rows

# ---------------------------------------------------------------- V2 free flight
h = build_human(g=9.81)
sysf = CoupledSystem(h, ContactParams(), lambda t, q, v: np.zeros(36), contact_enabled=False)
rng = np.random.default_rng(7)
q0 = rng.uniform(-0.3, 0.3, 36); q0[2] = 5.0
v0 = rng.uniform(-1.5, 1.5, 36)
ff = {}
for dt in (5e-4, 2.5e-4, 1.25e-4):
    y = sysf.pack(q0, v0)
    E0 = h.kinetic_energy(q0, v0) + h.potential_energy(q0)
    T0 = h.kinetic_energy(q0, v0)
    pin.computeCentroidalMomentum(h.model, h.data, q0, v0)
    L0 = h.data.hg.angular.copy()
    t = 0.0
    for _ in range(int(round(1.0 / dt))):
        y, _ = sysf.step(t, y, dt); t += dt
    E1 = h.kinetic_energy(y[:36], y[36:72]) + h.potential_energy(y[:36])
    pin.computeCentroidalMomentum(h.model, h.data, y[:36], y[36:72])
    ff[f"{dt*1e3:g}ms"] = dict(energy_drift_rel=float(abs(E1 - E0) / T0),
                               angular_momentum_drift_rel=float(np.abs(h.data.hg.angular - L0).max() / np.abs(L0).max()))
e = [ff[k]["energy_drift_rel"] for k in ("0.5ms", "0.25ms", "0.125ms")]
out["free_flight"] = dict(runs=ff, observed_order=[float(np.log2(e[0] / e[1])), float(np.log2(e[1] / e[2]))])

# --------------------------------------------------------------- V3 passive drop
pd = {}
for name, g in (("Earth", 9.81), ("Moon", 1.62)):
    hh = build_human(g=g)
    q = standing_posture(hh)
    kp, kd = np.full(30, 300.0), np.full(30, 8.0)
    ctrl = PostureController(q_ref=q.copy(), kp=kp, kd=kd, tau_ff=np.zeros(30), meta={})
    sy = CoupledSystem(hh, ContactParams(), ctrl, deck=deck)
    qq = q.copy(); qq[0], qq[1] = 7.0, 0.575; qq[2] += 0.02
    y = sy.pack(qq, np.zeros(36)); sy.reset_anchors(y)
    o = sy.simulate(y, T=0.6, dt=1e-4, out_dt=1e-3, record_q=True)
    E = o["energy"]; c = {k: i for i, k in enumerate(o["energy_cols"])}
    Esp = 0.5 * np.sum(kp * (o["q"][:, 6:] - q[6:]) ** 2, axis=1)
    through = (np.abs(np.diff(E[:, c["W_act"]])).sum() + np.abs(np.diff(E[:, c["W_contact"]])).sum()
               + E[-1, c["D_deck"]])
    total = E[:, c["T_h"]] + E[:, c["U_h"]] + E[:, c["E_deck"]] + E[:, c["E_contact"]] + Esp
    pd[name] = dict(balance_drift_J=float(np.abs(E[:, c["balance"]] - E[0, c["balance"]]).max()),
                    work_throughput_J=float(through),
                    balance_drift_rel=float(np.abs(E[:, c["balance"]] - E[0, c["balance"]]).max() / through),
                    total_energy_start_J=float(total[0]), total_energy_end_J=float(total[-1]),
                    dissipated_J=float(total[0] - total[-1]),
                    max_increase_between_samples_J=float(np.diff(total).max()),
                    max_increase_rel=float(np.diff(total).max() / (total[0] - total.min())),
                    min_normal_force_N=float(o["fn"].min()), peak_normal_force_over_W=float(o["fn"].sum(1).max() / (75 * g)),
                    max_penetration_mm=float(-1e3 * o["min_gap"].min()))
    if name == "Earth":
        curves["drop_t"] = o["t"]; curves["drop_total"] = total
        curves["drop_fn_over_W"] = o["fn"].sum(1) / (75 * g)
out["passive_drop"] = pd

# ---------------------------------------------------------- V4 time-step study
ms = make_multisine(9, T=8.0, rms=80.0, seed=5)
res = {}
for dt in (4e-4, 2e-4, 1e-4, 5e-5):
    sc = make_standing(g=9.81, deck=deck)
    t0 = time.perf_counter()
    res[dt] = sc.system.simulate(sc.y_eq.copy(), T=0.8, dt=dt, out_dt=2e-3, forcing=ms)
    res[dt]["wall"] = time.perf_counter() - t0
ref = res[5e-5]
conv = {}
for dt in (4e-4, 2e-4, 1e-4):
    ee = {}
    for nm_ in ("pelvis_L", "wrist_L"):
        k = ref["sensors"].index(nm_)
        x, xr = res[dt]["f"][:, k], ref["f"][:, k]
        ee[nm_] = float(np.sqrt(np.mean((x - xr) ** 2)) / np.sqrt(np.mean((xr - xr[0]) ** 2)))
    bal = res[dt]["energy"][:, 8]
    conv[f"{dt*1e3:g}ms"] = dict(rel_rms_error=ee, balance_drift_J=float(np.abs(bal - bal[0]).max()),
                                 wall_per_sim_second=float(res[dt]["wall"] / 0.8))
errs = [max(conv[k]["rel_rms_error"].values()) for k in ("0.4ms", "0.2ms", "0.1ms")]
out["time_step"] = dict(reference="0.05 ms", runs=conv,
                        observed_order=[float(np.log2(errs[0] / errs[1])), float(np.log2(errs[1] / errs[2]))])

# ------------------------------------ V5 linear model vs time-domain simulation
# single input: vertical force of the mid-span shaker, exactly periodic multisine
Tp, nper = 8.0, 5
sc = make_standing(g=9.81, deck=deck)
s, y = sc.system, sc.y_eq
A, BF, _ = s.linearize(y)
C, D, _, names = s.sensor_jacobians(y)
ms1 = make_multisine(9, T=Tp, rms=0.0, seed=21, f_min=0.25, f_flat=50.0, f_max=50.0, order=8)
one = make_multisine(1, T=Tp, rms=20.0, seed=21, f_min=0.25, f_flat=50.0, f_max=50.0, order=8)
ms1.amp[5], ms1.phase[5] = one.amp[0], one.phase[0]          # channel 5 = shaker 2 (mid-span), z
dt, odt = 2.5e-4, 2e-3
t0 = time.perf_counter()
o = s.simulate(y.copy(), T=nper * Tp, dt=dt, out_dt=odt, forcing=ms1)
wall = time.perf_counter() - t0
n1 = int(round(Tp / odt))
last = slice(len(o["t"]) - 1 - n1, len(o["t"]) - 1)
Fz = ms1.sample(odt, n1, t0=(nper - 1) * Tp)[5]
Ff = np.fft.rfft(Fz)
kk = one.k
fl = kk / Tp
Hlin = frf(A, BF[:, 5:6], C, D[:, 5:6], fl)[:, :, 0]
cmp_ = {}
for nm_, ch in (("pelvis_L", 2), ("pelvis_L", 1), ("pelvis_L", 0), ("wrist_L", 2), ("wrist_L", 0)):
    i = names.index(nm_)
    ysig = o["f"][last, i, ch]
    Ht = np.fft.rfft(ysig - ysig.mean())[kk] / Ff[kk]
    Hl = Hlin[:, 6 * i + ch]
    band = (fl >= 1.0)
    rel = np.abs(Ht - Hl) / np.abs(Hl)
    w = np.abs(Hl) ** 2
    cmp_[f"{nm_}_f{'xyz'[ch]}"] = dict(median_rel_err=float(np.median(rel[band])),
                                       energy_weighted_rel_err=float(np.sqrt(np.sum(w[band] * rel[band] ** 2) / np.sum(w[band]))),
                                       max_rel_err=float(rel[band].max()))
    if (nm_, ch) in (("pelvis_L", 2), ("wrist_L", 2)):
        curves[f"lin_{nm_}_z"] = np.abs(Hl); curves[f"sim_{nm_}_z"] = np.abs(Ht)
curves["lin_f"] = fl
out["linear_vs_time_domain"] = dict(
    description="Earth, S1 deck, mid-span vertical shaker only, periodic multisine 20 N RMS, "
                "5 periods of 8 s, last period analysed at the excited lines (no leakage)",
    comparisons=cmp_, wall_per_sim_second=float(wall / (nper * Tp)),
    sumFn_over_W_range=[float(o["fn"].sum(1).min() / sc.info["weight"]), float(o["fn"].sum(1).max() / sc.info["weight"])],
    n_slip=int(o["n_slip"]), balance_drift_J=float(np.abs(o["energy"][:, 8] - o["energy"][0, 8]).max()),
    work_ext_J=float(o["energy"][-1, 7]), deck_dissipation_J=float(o["energy"][-1, 6]))

# ------------------------------------------------------------------- V6 timing
sc = make_standing(g=9.81, deck=deck)
s, y = sc.system, sc.y_eq
F = np.zeros(9)
tim = {}
for label, flag in (("compiled", True), ("numpy_reference", False)):
    s.use_kernel = flag
    t0 = time.perf_counter()
    for _ in range(3000):
        s.rhs(0.0, y, F)
    tim[label + "_rhs_microseconds"] = float(1e6 * (time.perf_counter() - t0) / 3000)
s.use_kernel = True
out["timing"] = tim

out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
os.makedirs(os.path.join(ROOT, "outputs", "tables"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "outputs", "processed"), exist_ok=True)
with open(os.path.join(ROOT, "outputs", "tables", "verification_summary.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
np.savez_compressed(os.path.join(ROOT, "outputs", "processed", "verification_curves.npz"), **curves)
print("wrote outputs/tables/verification_summary.json")
