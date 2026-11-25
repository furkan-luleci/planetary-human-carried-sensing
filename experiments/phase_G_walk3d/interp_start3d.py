"""Continuation start for the upright 3-D walker: the sagittal parameters are moved part of
the way (alpha) from a walking 3-D gait (A) towards the upright planar Earth gait (B), the
frontal-plane parameters are kept from A; each mix is evaluated with the 3-D cost.

    python experiments/phase_G_walk3d/interp_start3d.py A.json [alpha ...]
writes outputs/gait/_start3d_alpha<alpha>.json for each mix (fields free, x01, cost, metrics).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import optimize_gait3d as og                     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
O = os.path.join(ROOT, "outputs", "gait")


def _eval(x):
    og._init(dict(g=og.G_E, v0=1.0, T=9.0, t_skip=4.0, leg="nominal_n1.5_constant",
                  targets={**og.COMMON, **og.EARTH_TARGETS[1.0]}))
    J, m = og.cost(np.array(x))
    return J, m


def main():
    A = json.load(open(sys.argv[1]))
    alphas = [float(a) for a in sys.argv[2:]] or [0.2, 0.4, 0.6]
    planar = json.load(open(os.path.join(O, "earth_1.00.json")))
    xa = np.array([A["x01"][A["free"].index(k)] for k in og.FREE])
    xb = xa.copy()
    for k, v in zip(planar["free"], planar["x01"]):
        xb[og.FREE.index(k)] = v
    X = [((1 - a) * xa + a * xb).tolist() for a in alphas]
    with Pool(min(len(X), 3)) as pool:
        R = pool.map(_eval, X)
    for a, x, (J, m) in zip(alphas, X, R):
        tl = m.get("trunk_lean")
        print(f"alpha {a:.2f}: cost {J:.2f} fell {m.get('fell')} trunk {np.degrees(tl) if tl is not None else float('nan'):.1f} deg "
              f"speed {m.get('speed', float('nan')):.2f}", flush=True)
        json.dump(dict(free=og.FREE, x01=x, cost=float(J), alpha=a, metrics=m, source_A=sys.argv[1], source_B="outputs/gait/earth_1.00.json"),
                  open(os.path.join(O, f"_start3d_alpha{a:.2f}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
