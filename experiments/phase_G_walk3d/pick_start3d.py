"""Starting point for re-tuning the 3-D walker with the upright-trunk bound.

Two candidates are evaluated with the 3-D cost (9 s, same targets as the tuning):
  (a) the previous 3-D result (outputs/gait/_leaning_v1/earth3d_1.00.json);
  (b) the sagittal parameters of the re-tuned (upright) planar Earth gait with the
      frontal-plane parameters of (a).
The cheaper one is written to outputs/gait/_start3d.json (fields free, x01, cost, which).

    python experiments/phase_G_walk3d/pick_start3d.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import optimize_gait3d as og                     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
O = os.path.join(ROOT, "outputs", "gait")


def main():
    og._init(dict(g=og.G_E, v0=1.0, T=9.0, t_skip=4.0, leg="nominal_n1.5_constant",
                  targets={**og.COMMON, **og.EARTH_TARGETS[1.0]}))
    old = json.load(open(os.path.join(O, "_leaning_v1", "earth3d_1.00.json")))
    planar = json.load(open(os.path.join(O, "earth_1.00.json")))
    xa = np.array([old["x01"][old["free"].index(k)] for k in og.FREE])
    xb = xa.copy()
    for k, v in zip(planar["free"], planar["x01"]):      # same bounds for the sagittal parameters
        xb[og.FREE.index(k)] = v
    res = []
    for which, x in (("previous 3-D gait", xa), ("upright planar gait + previous frontal-plane parameters", xb)):
        J, m = og.cost(x)
        res.append((J, which, x, m))
        print(f"{which}: cost {J:.2f}, trunk lean {m.get('trunk_lean', float('nan')):.3f} rad, fell {m.get('fell')}", flush=True)
    J, which, x, m = min(res, key=lambda r: r[0])
    json.dump(dict(free=og.FREE, x01=[float(v) for v in x], cost=float(J), which=which), open(os.path.join(O, "_start3d.json"), "w"), indent=1)
    print("start:", which)


if __name__ == "__main__":
    main()
