import numpy as np
import pinocchio as pin
import pytest

from planetsense.contact.compliant import ContactParams
from planetsense.coupling.system import CoupledSystem
from planetsense.human.control import PostureController, standing_posture
from planetsense.human.model import build_human
from tolerances import TOL


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_free_flight_conserves_energy_and_momentum(g, rng):
    """No contact, no torques: an articulated body tumbling under gravity.
    T + U is conserved, the angular momentum about the centre of mass is
    conserved, and the linear momentum changes by exactly m*g*t.  This probes
    the Coriolis / gyroscopic terms and the integrator."""
    h = build_human(g=g)
    sys = CoupledSystem(h, ContactParams(), lambda t, q, v: np.zeros(36), contact_enabled=False)
    q0 = rng.uniform(-0.3, 0.3, 36)
    q0[2] = 5.0
    v0 = rng.uniform(-1.5, 1.5, 36)
    y = sys.pack(q0, v0)
    E0 = h.kinetic_energy(q0, v0) + h.potential_energy(q0)
    T0 = h.kinetic_energy(q0, v0)
    pin.computeCentroidalMomentum(h.model, h.data, q0, v0)
    L0, P0 = h.data.hg.angular.copy(), h.data.hg.linear.copy()
    dt, n = 2.5e-4, 4000
    t = 0.0
    for _ in range(n):
        y, _ = sys.step(t, y, dt)
        t += dt
    q1, v1 = y[:36], y[36:72]
    E1 = h.kinetic_energy(q1, v1) + h.potential_energy(q1)
    pin.computeCentroidalMomentum(h.model, h.data, q1, v1)
    assert abs(E1 - E0) / T0 < TOL["energy_free_flight_rel"]
    assert np.abs(h.data.hg.angular - L0).max() / np.abs(L0).max() < TOL["momentum_free_flight_rel"]
    dP = h.data.hg.linear - P0
    expect = np.array([0, 0, -h.total_mass * g * t])
    assert np.abs(dP - expect).max() / abs(expect[2]) < TOL["momentum_free_flight_rel"]
    assert abs(y[-4]) == 0.0 and abs(y[-3]) == 0.0          # no actuator work, no contact work


def _passive_drop(g, deck, dt, T=0.6, height=0.02):
    h = build_human(g=g)
    q = standing_posture(h)
    kp, kd = np.full(30, 300.0), np.full(30, 8.0)
    ctrl = PostureController(q_ref=q.copy(), kp=kp, kd=kd, tau_ff=np.zeros(30), meta={})
    sys = CoupledSystem(h, ContactParams(), ctrl, deck=deck)
    q0 = q.copy()
    q0[0], q0[1] = 7.0, 0.575
    q0[2] += height
    y = sys.pack(q0, np.zeros(36))
    sys.reset_anchors(y)
    out = sys.simulate(y, T=T, dt=dt, out_dt=1e-3, record_q=True, fall_height=None)
    Esp = 0.5 * np.sum(kp * (out["q"][:, 6:] - q[6:]) ** 2, axis=1)
    return sys, out, Esp


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_energy_balance_passive_case(g, modal_deck):
    """Brief 17.5 / 15.1.  A passive body (joint springs and dampers only, no
    feed-forward, no feedback) is dropped onto the flexible deck.

    (a) Work-energy theorem: T_h + U_h + E_deck changes by exactly the work of
        joint torques and contact forces minus deck damping.  A wrong sign or a
        non-conjugate force mapping at the interface breaks this identity.
    (b) No energy creation: the total including the recoverable spring and
        contact energies never increases."""
    sys, out, Esp = _passive_drop(g, modal_deck, dt=1e-4)
    E = out["energy"]
    c = {k: i for i, k in enumerate(out["energy_cols"])}
    assert out["fn"].max() > 0 and out["fn"].min() >= 0.0          # it did land, never pulled
    through = (np.abs(np.diff(E[:, c["W_act"]])).sum() + np.abs(np.diff(E[:, c["W_contact"]])).sum()
               + E[-1, c["D_deck"]])
    drift = np.abs(E[:, c["balance"]] - E[0, c["balance"]]).max()
    assert drift / through < TOL["energy_balance_rel"], (drift, through)
    total = E[:, c["T_h"]] + E[:, c["U_h"]] + E[:, c["E_deck"]] + E[:, c["E_contact"]] + Esp
    scale = total[0] - total.min()
    assert scale > 0
    assert np.diff(total).max() < TOL["energy_creation_rel"] * scale, np.diff(total).max()
    assert total[-1] < total[0]                                     # and something was dissipated


def test_contact_unilateral_in_simulation(modal_deck):
    """Brief 17.5: during a drop-and-rebound the foot is never pulled down and
    the penetration stays small."""
    sys, out, _ = _passive_drop(9.81, modal_deck, dt=1e-4, height=0.03)
    assert out["fn"].min() >= 0.0
    opened = out["min_gap"] > 0
    assert opened.any()                                             # airborne at the start
    assert (out["fn"][out["min_gap"] > 1e-6].min(axis=1) == 0).all()
    assert -out["min_gap"].min() < 0.02                             # < 2 cm even at impact
