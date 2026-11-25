"""Phase K: the first mode of each structure with a person standing at its standing point,
every body variant of Phase D, each planet (linear model): frequency and damping of the peak
of the driving-point response under the feet nearest the empty first mode.
Writes outputs/tables/hsi_structures.json.

    PYTHONPATH=src python experiments/phase_K_S2/hsi_bodies.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_K_scope_S2"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scope_s2 import G, peaks  # noqa: E402
from scope_s2_variants import VARIANTS, occupied_variant  # noqa: E402
from planetsense.structure import registry as SR  # noqa: E402

out = dict(method="linear, standing person at the structure's standing point; peak of the driving-point vertical "
                  "response under the feet nearest the empty first mode; damping from the half-power width", data={})
for name in ("S1", "S2", "S2L"):
    md, st = SR.modal_deck(name), SR.get(name)
    fb1 = float(md.freq_hz[0])
    f = np.linspace(0.6 * fb1, 1.4 * fb1, 4001)
    d = dict(empty=dict(f=fb1, zeta=float(md.zeta[0])), standing_point=list(st.stand_xy), bodies={})
    for v in VARIANTS:
        d["bodies"][v] = {}
        for p, g in G.items():
            H = occupied_variant(md, g, st.stand_xy[0], st.stand_xy[1], f, v)
            pk = peaks(f, H, 0.6 * fb1, 1.4 * fb1)
            q = min(pk, key=lambda r: abs(r["f"] - fb1)) if pk else None
            d["bodies"][v][p] = None if q is None else dict(f=q["f"], zeta=q["zeta_half_power"])
    out["data"][name] = d
    print(name, {p: (round(d["bodies"]["nominal"][p]["f"], 3), round(100 * d["bodies"]["nominal"][p]["zeta"], 2)) for p in G}, flush=True)
json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "hsi_structures.json"), "w"), indent=1)
