"""Phase K, before the trials: the frequency of every mode below 30 Hz of S2 and S2L with a
person standing at the standing point (nominal calibrated body), each planet; linear model,
peak of the driving-point response under the feet nearest each empty mode (within 15 %).
These are the 'occupied' frequencies of the declared recovery rule
(docs/phase_reports/PHASE_K_declared_rules.md).  Writes outputs/tables/occupied_modes.json.

    PYTHONPATH=src python experiments/phase_K_scope_S2/occupied_modes.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from scope_s2 import G, peaks  # noqa: E402
from scope_s2_variants import occupied_variant  # noqa: E402
from planetsense.structure import registry as SR  # noqa: E402

out = dict(method="linear model, nominal calibrated body standing at the structure's standing point; peak of the "
                  "driving-point vertical response under the feet nearest each empty mode (within 15 %)", data={})
for name in ("S2", "S2L"):
    md, st = SR.modal_deck(name), SR.get(name)
    f = np.linspace(1.0, 32.0, 6201)
    modes = [(l, float(q)) for l, q in zip(md.labels, md.freq_hz) if q < 30.0]
    out["data"][name] = dict(empty={l: q for l, q in modes}, standing_point=list(st.stand_xy), occupied={})
    for p, g in G.items():
        H = occupied_variant(md, g, st.stand_xy[0], st.stand_xy[1], f, "nominal")
        pk = peaks(f, H, 0.8, 31.0)
        occ = {}
        for l, q in modes:
            near = [r for r in pk if abs(r["f"] - q) <= 0.15 * q]
            occ[l] = None if not near else dict(f=min(near, key=lambda r: abs(r["f"] - q))["f"],
                                               zeta=min(near, key=lambda r: abs(r["f"] - q))["zeta_half_power"])
        out["data"][name]["occupied"][p] = occ
        print(name, p, {l: (round(v["f"], 3), None if v["zeta"] is None else round(100 * v["zeta"], 1)) if v else None
                        for l, v in occ.items()}, "| empty", {l: round(q, 3) for l, q in modes}, flush=True)
json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "occupied_modes.json"), "w"), indent=1)
