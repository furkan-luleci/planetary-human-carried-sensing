"""Counterfactual gaits: the Earth controller transferred to Mars and Moon by
dynamic similarity (equal Froude number, Fr = v^2 / (g L), L = 0.93 m).

Times are stretched by sqrt(g_E / g), joint gains and torques follow the local
weight, the speed is v_E sqrt(g / g_E).  The passive tissue (leg springs, foot
pads) is NOT rescaled, so the similarity is approximate.  Published harness
experiments show that people do not walk like this in reduced gravity; these
cases are labelled "what-if" everywhere they are used.

    python experiments/phase_C_walking/similarity_cases.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from optimize_gait import G_E                                          # noqa: E402
from planetsense.human.body_v2 import load_leg                               # noqa: E402
from planetsense.simulation.walking import WalkParams, gait_metrics, make_walker   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
O = os.path.join(ROOT, "outputs", "gait")
src = json.load(open(os.path.join(O, "earth_1.25.json")))
pE = WalkParams(**src["params"])
vE = src["metrics_validation_30s"]["speed"]
for name, g in (("mars_0.77_similar", 3.71), ("moon_0.51_similar", 1.62)):
    out_path = os.path.join(O, name + ".json")
    if os.path.exists(out_path):
        continue
    p = pE.similarity_scaled(g, G_E)
    v = vE * np.sqrt(g / G_E)
    sys_, y = make_walker(g, p, load_leg(), v0=v, g_gain=g)
    out = sys_.simulate(y, T=40.0, dt=2.5e-4, out_dt=5e-3, fall_height=0.6, record_q=True)
    m = gait_metrics(out, sys_.h, t_skip=12.0, i_act=sys_.controller.i_act)
    if "dlf" in m:
        m["dlf1"] = m["dlf"][0]
    res = dict(kind="similarity", g=g, g_gain=g, target_speed=float(v), source="earth_1.25",
               froude=float(vE ** 2 / (G_E * 0.93)), params=p.asdict(), metrics_validation_30s=m,
               note="Earth 1.25 m/s controller, times x sqrt(g_E/g), gains x g/g_E; passive tissue unchanged")
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print(name, "fell" if m.get("fell") else "walks", {k: round(m[k], 3) for k in ("speed", "step_frequency", "duty_factor",
          "flight_fraction", "grf_foot_peak_over_W") if k in m}, flush=True)
