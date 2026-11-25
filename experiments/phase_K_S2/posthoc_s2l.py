"""Phase K, post hoc (run after the declared criteria were evaluated; changes no verdict):
why S2L's first mode is missed on Earth while standing, and why criterion 4 fails there;
the same quantities for S2, for comparison.
(1) prominence of the first-mode peak in the phone's (thigh-pocket) ensemble spectrum over its
median within +/-2 Hz (the frozen rule needs 4 dB over a running median), peak searched between
0.8 and 1.4 times the empty first mode; (2) the linear model driven by the actual shaker layout:
whether the floor's vertical acceleration PSD under the feet has an interior maximum between 0.75
and 1.35 times the empty first mode, and where.  Writes outputs/tables/phaseK_posthoc_<s>.json.

    PYTHONPATH=src python experiments/phase_K_S2/posthoc_s2l.py [S2L|S2]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_F_outcomes"))
sys.path.insert(0, os.path.join(ROOT, "src"))
STRUCT = sys.argv[1] if len(sys.argv) > 1 else "S2L"
os.environ["PLANETSENSE_STRUCTURE"] = STRUCT
import outcomes as OC  # noqa: E402
from planetsense.evaluation.linear import frf  # noqa: E402
from planetsense.human.body_v2 import load_leg  # noqa: E402
from planetsense.simulation.standing import make_standing  # noqa: E402
from planetsense.structure import registry as SR  # noqa: E402

md, st = SR.modal_deck(STRUCT), SR.get(STRUCT)
fb1 = float(md.freq_hz[0])
res = {}
for p, g in (("Earth", 9.81), ("Mars", 3.71), ("Moon", 1.62)):
    recs = OC.load_group("standing", "E1", p, None)
    f, Sn = OC.spectra_of(recs, 16.0, True, seed0=1)
    S = Sn["thigh"]
    k = (f > 0.8 * fb1) & (f < 1.4 * fb1)
    j = int(np.argmax(np.where(k, S, 0)))
    med = np.median(S[(f > f[j] - 2) & (f < f[j] + 2)])
    sc = make_standing(g=g, deck=md, leg=load_leg(), x_com=st.stand_xy[0], y_com=st.stand_xy[1], shakers=True)
    s = sc.system
    A, BF, _ = s.linearize(sc.y_eq)
    nq, nm, nx = s.nq, s.nm, A.shape[0]
    C = np.zeros((1, nx)); C[0, 2 * nq:2 * nq + nm] = md.surface_modes([st.stand_xy[0]], [st.stand_xy[1]])[0, 2, :]
    ff = np.linspace(0.75 * fb1, 1.35 * fb1, 1201)
    H = frf(A, BF, C, np.zeros((1, BF.shape[1])), ff)[:, 0, :] * (-(2 * np.pi * ff) ** 2)[:, None]
    P = (np.abs(H) ** 2).sum(axis=1)
    i = int(np.argmax(P))
    res[p] = dict(device_peak_hz=float(f[j]), device_peak_over_median_db=float(10 * np.log10(S[j] / med)),
                  floor_psd_interior_peak=bool(0 < i < len(ff) - 1),
                  floor_psd_peak_linear_shakers_hz=float(ff[i]) if 0 < i < len(ff) - 1 else None)
    print(p, res[p], flush=True)
json.dump(dict(note="post hoc, after the declared criteria were evaluated; does not change any verdict", data=res),
          open(os.path.join(ROOT, "outputs", "tables", f"phaseK_posthoc_{STRUCT.lower()}.json"), "w"), indent=1)
