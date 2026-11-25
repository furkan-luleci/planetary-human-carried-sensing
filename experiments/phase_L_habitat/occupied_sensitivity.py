"""Phase L (2026-11-22): sensitivity of the habitat floor's occupied first mode to its two weakest
assumptions, the effective width of floor that moves with one beam line (halved, as derived, doubled)
and the floor's damping (0.5, 1 and 2 %), with the method and standing point of the study
(experiments/phase_K_S2/hsi_bodies.py): linear model, person at the standing point (1.64 m from the
wall), peak of the driving-point vertical response under the feet nearest the empty first mode,
damping from the half-power width; nominal body and the control body in which nothing follows the
load.  The pre-run check (experiments/phase_K_scope_S2/prerun_checks.py) placed the person at the
middle of the first span; this repeats it at the study's standing point.

Writes outputs/tables/occupied_sensitivity_S2.json.

    python experiments/phase_L_habitat/occupied_sensitivity.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_K_scope_S2"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scope_s2 import G, peaks  # noqa: E402
from scope_s2_variants import occupied_variant  # noqa: E402
from planetsense.structure import registry as SR  # noqa: E402
from planetsense.structure.habitat_deck import HabitatDeck, build  # noqa: E402


def first_mode(md, g, x, y, variant):
    fb1 = float(md.freq_hz[0])
    f = np.linspace(0.6 * fb1, 1.4 * fb1, 4001)
    H = occupied_variant(md, g, x, y, f, variant)
    pk = peaks(f, H, 0.6 * fb1, 1.4 * fb1)
    q = min(pk, key=lambda r: abs(r["f"] - fb1)) if pk else None
    return None if q is None else dict(f=q["f"], zeta=q["zeta_half_power"])


def main():
    x, y = SR.get("S2").stand_xy
    ref = SR.modal_deck("S2")
    rows = []
    for bf in (0.5, 1.0, 2.0):
        for z in (0.005, 0.01, 0.02):
            h = HabitatDeck(B_factor=bf, zeta=z)
            md = build(h).modal(150.0)
            row = dict(B_factor=bf, zeta_floor=z, empty_B1=float(md.freq_hz[0]), strip_mass=float(h.design()["total_mass_strip"]))
            for body in ("nominal", "fully_linear"):
                row[body] = {p: first_mode(md, g, x, y, body) for p, g in G.items()}
            rows.append(row)
            print(f"width x{bf:<3} floor damping {100 * z:.1f} %: empty {row['empty_B1']:.2f} Hz | nominal "
                  + "  ".join(f"{p} {row['nominal'][p]['f']:.2f} Hz {100 * row['nominal'][p]['zeta']:.1f} %" for p in G)
                  + " | control " + "  ".join(f"{p} {100 * row['fully_linear'][p]['zeta']:.1f} %" for p in G), flush=True)
    base = next(r for r in rows if r["B_factor"] == 1.0 and r["zeta_floor"] == 0.01)
    assert abs(base["empty_B1"] - float(ref.freq_hz[0])) < 1e-6, "the nominal case must be the study's floor"
    out = dict(status="SIMULATION - linear model; post hoc sensitivity at the study's standing point",
               standing_point=[x, y], method="as experiments/phase_K_S2/hsi_bodies.py", rows=rows,
               falls_with_gravity_nominal=all(r["nominal"]["Earth"]["zeta"] > r["nominal"]["Mars"]["zeta"] > r["nominal"]["Moon"]["zeta"]
                                               for r in rows))
    json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "occupied_sensitivity_S2.json"), "w"), indent=1, default=float)
    print("falls with gravity in every case (nominal body):", out["falls_with_gravity_nominal"])


if __name__ == "__main__":
    main()
