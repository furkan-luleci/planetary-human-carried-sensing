"""Phase K scope: how much the occupied first mode of S2 depends on the body assumptions.
Every standing-body variant of Phase D (same definitions as
experiments/phase_D_standing/standing_v2.py), each planet, linearised: the peak of the
driving-point response under the feet near B1 and its half-power damping.
Adds "hsi_variants" to outputs/tables/scope_s2.json.

    PYTHONPATH=src python experiments/phase_K_scope_S2/scope_s2_variants.py
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
from scope_s2 import G, occupied, peaks  # noqa: E402
from planetsense.contact.compliant import ContactParams, tangent_stiffness  # noqa: E402
from planetsense.human.body_v2 import load_leg  # noqa: E402
from planetsense.simulation.standing import make_standing  # noqa: E402

nominal_pad = ContactParams()
linear_pad = ContactParams(k_n=tangent_stiffness(nominal_pad, 75.0 * 9.81 / 8), n_exp=1.0,
                           c_n=nominal_pad.c_n, k_t=nominal_pad.k_t, c_t=nominal_pad.c_t, mu=nominal_pad.mu)
VARIANTS = {                                   # as in experiments/phase_D_standing/standing_v2.py
    "nominal": dict(leg="nominal_n1.5_constant"),
    "earth_fixed": dict(leg="nominal_n1.5_constant", gain_policy="earth_fixed"),
    "leg_n1": dict(leg="nominal_n1_constant"),
    "leg_n2": dict(leg="nominal_n2_constant"),
    "soft_damped": dict(leg="soft_damped_n1.5_constant"),
    "stiff_light": dict(leg="stiff_light_n1.5_constant"),
    "prop_damping": dict(leg="nominal_n1.5_proportional"),
    "fully_linear": dict(leg="nominal_n1_constant", gain_policy="earth_fixed", contact=linear_pad),
    "rigid_leg": dict(leg=None),
}


def occupied_variant(md, g, x, y, f, v):
    kw = dict(VARIANTS[v])
    legname = kw.pop("leg")
    sc = make_standing(g=g, deck=md, leg=load_leg(legname) if legname else None, x_com=x, y_com=y, shakers=False, **kw)
    s = sc.system
    A, _, f0 = s.linearize(sc.y_eq)
    from planetsense.evaluation.linear import frf
    nq, nm, nx = s.nq, s.nm, A.shape[0]
    b = md.surface_modes([x], [y])[0, 2, :]
    B = np.zeros((nx, 1)); B[2 * nq + nm:2 * nq + 2 * nm, 0] = b
    C = np.zeros((1, nx)); C[0, 2 * nq:2 * nq + nm] = b
    return frf(A, B, C, np.zeros((1, 1)), f)[:, 0, 0] * -(2 * np.pi * f) ** 2


def main():
    path = os.path.join(ROOT, "outputs", "tables", "scope_s2.json")
    sc = json.load(open(path))
    md = S2.build(S2.HabitatDeck()).modal(f_max=150.0)
    x, y = sc["standing_point"]
    f = np.linspace(4.0, 25.0, 4201)
    fb1 = md.freq_hz[0]
    res = {}
    for v in VARIANTS:
        res[v] = {}
        for p, g in G.items():
            H = occupied_variant(md, g, x, y, f, v)
            pk = peaks(f, H, 0.6 * fb1, 1.3 * fb1)
            top = min(pk, key=lambda q: abs(q["f"] - fb1)) if pk else None      # the peak of the first mode
            res[v][p] = top
        print(f"{v:13s} " + "  ".join(f"{p}: {res[v][p]['f']:.2f} Hz, zeta {100 * (res[v][p]['zeta_half_power'] or float('nan')):.1f} %"
                                       for p in G), flush=True)
    sc["hsi_variants"] = dict(note="peak of the driving-point response under the feet near B1, linearised, per body variant",
                              empty_B1=float(fb1), results=res)
    json.dump(sc, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
