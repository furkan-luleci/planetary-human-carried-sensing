"""Calibrate the passive leg compliance to published summary values of the
standing body's vertical apparent mass (Earth gravity).

Targets (all read at abstract level; see docs/literature/C_hsi_prior_work_novelty.md,
section 2, rows 1, 10, 11):
  nominal     resonance 5.5 Hz (row 1); damping 0.30 = midpoint of rows 10 and 11
  soft-damped equivalent single-degree-of-freedom 5.24 Hz, 39 % (row 10)
  stiff-light equivalent single-degree-of-freedom 6.6 Hz, 22 %  (row 11)

Two numbers are fitted (leg stiffness and damping) to two numbers (frequency
and damping ratio of the single-degree-of-freedom system that best matches the
model's apparent mass between 0.5 and 20 Hz).  This is a calibration of the
model class to summary statistics of other people's subjects, not a fit to any
individual and not a validation.  The stiffening exponent n and the damping
form cannot be identified from these targets and are carried as assumptions.

    python experiments/phase_B_body/calibrate_leg.py
"""
import json
import os
import subprocess
import time

import numpy as np
from scipy.optimize import least_squares

from planetsense.evaluation.linear import apparent_mass, fit_sdof, heavy_platform
from planetsense.human.control import LegSpring
from planetsense.simulation.standing import make_standing

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FR = np.linspace(0.5, 20.0, 100)
PLAT = heavy_platform()
F0 = 0.5 * 73.0 * 9.81            # approximate static load of one leg (N), for scaling only

TARGETS = {"nominal": (5.5, 0.30), "soft_damped": (5.24, 0.39), "stiff_light": (6.6, 0.22)}
CASES = [("nominal", 1.5, "constant"), ("nominal", 1.0, "constant"), ("nominal", 2.0, "constant"),
         ("soft_damped", 1.5, "constant"), ("stiff_light", 1.5, "constant"),
         ("nominal", 1.5, "proportional")]


def leg_from(kt, c, n, damping):
    d0 = n * F0 / kt                      # compression giving tangent stiffness kt at load F0
    return LegSpring(k=F0 / d0 ** n, n=n, c=c, damping=damping)


def measure(leg, g=9.81):
    sc = make_standing(g=g, deck=PLAT, leg=leg)
    M = apparent_mass(sc, FR) / sc.human.total_mass
    f, z = fit_sdof(FR, M)
    return f, z, float(np.abs(M).max()), float(FR[np.abs(M).argmax()]), sc


out = dict(targets=TARGETS, fit_band_hz=[0.5, 20.0], started=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           commit_at_run=subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           sets={})
for tname, n, damping in CASES:
    ft, zt = TARGETS[tname]
    c0 = 1500.0 if damping == "constant" else 1500.0 / F0

    def res(p):
        leg = leg_from(p[0] * 7e4, p[1] * c0, n, damping)
        f, z, *_ = measure(leg)
        return [(f - ft) / ft, (z - zt) / zt]

    sol = least_squares(res, [1.0, 1.0], bounds=([0.2, 0.1], [6.0, 8.0]), xtol=1e-6, ftol=1e-10, diff_step=1e-3)
    leg = leg_from(sol.x[0] * 7e4, sol.x[1] * c0, n, damping)
    f, z, pk, fpk, sc = measure(leg)
    key = f"{tname}_n{n:g}_{damping}"
    out["sets"][key] = dict(target=tname, target_f=ft, target_zeta=zt, n=n, damping=damping,
                            k=leg.k, c=leg.c, k_ext=leg.k_ext, c_ext=leg.c_ext,
                            fitted_f=f, fitted_zeta=z, peak_normalised_apparent_mass=pk, peak_f=fpk,
                            static_leg_compression_mm=float(1e3 * sc.info["leg_compression"].mean()),
                            tangent_stiffness_per_leg_kN_m=float(1e-3 * leg.tangent(F0)),
                            residual=[float(r) for r in sol.fun], nfev=int(sol.nfev))
    print(key, "k=%.4g c=%.4g -> f=%.3f zeta=%.3f peak %.2f at %.2f Hz, leg compression %.1f mm" % (
        leg.k, leg.c, f, z, pk, fpk, out["sets"][key]["static_leg_compression_mm"]), flush=True)
with open(os.path.join(ROOT, "configs", "body_v2_leg_calibration.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote configs/body_v2_leg_calibration.json")
