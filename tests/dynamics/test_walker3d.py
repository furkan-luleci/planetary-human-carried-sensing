"""Mechanics of the 3-D walker (Phase G) on the flexible deck."""
import numpy as np
import pinocchio as pin
import pytest

from planetsense.human.body_v2 import load_leg
from planetsense.simulation.walking3d import WalkParams3D, make_walker3d
from tests.tolerances import TOL


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_walker3d_mechanics_are_consistent(g, modal_deck):
    """The 3-D walker walking off the centreline of the flexible deck: contact
    never pulls; the centre of mass obeys m a = sum f + m g (+ the start-up aid's
    force on the pelvis, an external force) in all three directions at every
    sampled state; the work-energy identity holds (no energy is created by the
    coupling, whatever the controller does)."""
    # non-default controller options exercise every code path used by the tuned walker
    p = WalkParams3D(tau_head=1.0, lam_lat=0.5, k_yaw=3.5, brake=50.0, zab=1.2, lumbar_ext=0.05)
    s, y = make_walker3d(g, p, load_leg(), deck=modal_deck, x0=4.0, y_walk=0.575, v0=0.5)
    h, c = s.h, s.controller
    ir = [c.iroot[k] for k in ("x", "y", "z")]
    m_tot = h.total_mass
    dt, t, worst = 2.5e-4, 0.0, 0.0
    E0 = s.energy(y)
    through, last, n_contact = 0.0, np.zeros(4), 0
    for i in range(int(1.5 / dt)):
        if i % 100 == 0:
            k1 = s.rhs(t, y, None, first=True)
            q, v, a = y[:s.nq], y[s.nq:2 * s.nq], k1[s.nq:2 * s.nq]
            assert s._fn.min() >= 0.0
            n_contact += int(s._fn.sum() > 0)
            pin.centerOfMass(h.model, h.data, q, v, a)
            lhs = m_tot * h.data.acom[0]
            rhs = s._f.sum(axis=0) + m_tot * np.array([0, 0, -g]) + np.asarray(s.last["tau"])[ir]
            worst = max(worst, np.abs(lhs - rhs).max() / (m_tot * g))
            W = y[-4:]
            through += np.abs(W - last).sum()
            last = W.copy()
        y, _ = s.step(t, y, dt)
        t += dt
        if y[s.iz] < 0.5:
            break
    assert n_contact > 10
    assert worst < TOL["walk3d_centroidal_rel"], worst
    drift = abs(s.energy(y)["balance"] - E0["balance"])
    assert drift / through < TOL["walk3d_energy_balance_rel"], (drift, through)
