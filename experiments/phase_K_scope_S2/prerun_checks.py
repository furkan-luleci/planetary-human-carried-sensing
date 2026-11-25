"""Phase K, checks before any S2 trial (2026-11-21), linear model, standing person:

1. Sensitivity of S2's occupied first mode to the two weakest assumptions: the width of
   floor that moves with one beam line (x0.5, x1, x2 the Design Guide 11 value) and the
   floor's own damping (0.5, 1, 2 %).  Nominal body and the control body (nothing in the
   body follows the load).
2. Design gravity: the same floor layout (two 4.10 m spans) with members sized for Mars
   weight or for Moon weight instead of the Earth-tested HDU criterion (S2L).
   Every calibrated body variant, each planet.

Writes outputs/tables/prerun_s2.json.

    PYTHONPATH=src python experiments/phase_K_scope_S2/prerun_checks.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import s2_deck as S2  # noqa: E402
from scope_s2 import G, peaks  # noqa: E402
from scope_s2_variants import VARIANTS, occupied_variant  # noqa: E402

CAL = [v for v in VARIANTS if v not in ("fully_linear", "rigid_leg")]


def first_mode(md, g, variant, f):
    x, y = 0.25 * md.params.L, 0.0
    H = occupied_variant(md, g, x, y, f, variant)
    fb1 = md.freq_hz[0]
    pk = peaks(f, H, 0.6 * fb1, 1.3 * fb1)
    q = min(pk, key=lambda r: abs(r["f"] - fb1)) if pk else None
    return None if q is None else dict(f=q["f"], zeta=q["zeta_half_power"])


def main():
    out = dict(status="linear checks before the S2 trials; standing person at the middle of the first span")
    sens = []
    for bf in (0.5, 1.0, 2.0):
        for z in (0.005, 0.01, 0.02):
            h = S2.HabitatDeck(B_factor=bf, zeta=z)
            md = S2.build(h).modal(150.0)
            f = np.linspace(0.4 * md.freq_hz[0], 1.6 * md.freq_hz[0], 3001)
            row = dict(B_factor=bf, zeta_floor=z, empty_B1=float(md.freq_hz[0]), strip_mass=h.design()["total_mass_strip"])
            for body in ("nominal", "fully_linear"):
                row[body] = {p: first_mode(md, g, body, f) for p, g in G.items()}
            sens.append(row)
            print(f"width x{bf:<3} floor damping {100*z:.1f} %: empty {row['empty_B1']:.2f} Hz | nominal " +
                  "  ".join(f"{p} {row['nominal'][p]['f']:.2f} Hz {100*row['nominal'][p]['zeta']:.1f} %" for p in G) +
                  " | control " + "  ".join(f"{p} {100*row['fully_linear'][p]['zeta']:.1f} %" for p in G), flush=True)
    out["sensitivity"] = sens
    var = {}
    for name, gr in (("S2", 1.0), ("S2_mars_weight", 3.71 / 9.81), ("S2L_moon_weight", 1.62 / 9.81)):
        h = S2.HabitatDeck(g_ratio=gr, n_spans=2)
        d = h.design()
        md = S2.build(h).modal(150.0)
        f = np.linspace(0.4 * md.freq_hz[0], 1.6 * md.freq_hz[0], 3001)
        v = dict(g_ratio=gr, beam=d["beam"], floor_kg_m2=d["m_area"], B_eff=d["B_eff"], strip_mass=d["total_mass_strip"],
                 modes=[dict(label=l, f=float(q)) for l, q in zip(md.labels, md.freq_hz) if q < 35],
                 min_section=bool(d["beam"]["bf"] <= 0.0401 and d["beam"]["tf"] <= 0.00201), bodies={})
        for body in list(VARIANTS):
            v["bodies"][body] = {p: first_mode(md, g, body, f) for p, g in G.items()}
        var[name] = v
        cal = lambda p, k: [v["bodies"][b][p][k] for b in CAL if v["bodies"][b][p]]
        print(f"{name:16s} B1 {md.freq_hz[0]:.2f} Hz, floor {d['m_area']:.1f} kg/m2, min section {v['min_section']} | calibrated bodies: " +
              "  ".join(f"{p} {min(cal(p,'f')):.2f}-{max(cal(p,'f')):.2f} Hz, {100*min(cal(p,'zeta')):.1f}-{100*max(cal(p,'zeta')):.1f} %" for p in G) +
              " | control " + "  ".join(f"{p} {100*v['bodies']['fully_linear'][p]['zeta']:.1f} %" for p in G), flush=True)
    out["design_gravity"] = var
    json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "prerun_s2.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
