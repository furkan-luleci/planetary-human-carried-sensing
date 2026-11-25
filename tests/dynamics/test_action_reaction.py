import numpy as np
import pinocchio as pin
import pytest

from planetsense.simulation.standing import make_standing
from tolerances import TOL


def _points(s, q):
    pin.forwardKinematics(s.h.model, s.h.data, q, np.zeros(s.nq))
    return s._feet(q, np.zeros(s.nq))[0].copy()


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_contact_force_action_reaction(g, modal_deck, rng):
    """Brief 17.5 / 8.2.  For arbitrary states with sliding, lifting and loaded
    points, the generalised forces that the contact puts on the two bodies must
    satisfy the virtual-work identity

        tau_c . dv + Q_c . d(eta_dot) = sum_i f_i . (dv_point,i - dv_surface,i)

    for every virtual velocity.  tau_c and Q_c are read from the dynamics
    (inverse dynamics with/without contact, deck equation with/without
    contact); point velocities come from finite differences of positions, so
    the check does not reuse the code path that applied the forces."""
    sc = make_standing(g=g, deck=modal_deck)
    s, y0, h = sc.system, sc.y_eq, sc.human
    nq, nm = s.nq, s.nm
    scale = np.concatenate([np.full(36, 4e-3), np.full(36, 8e-2), np.full(nm, 2e-4),
                            np.full(nm, 2e-2), np.full(s.nc, 1e-3)])
    worst = 0.0
    for trial in range(25):
        y = y0.copy()
        y[:s.nx] += rng.normal(0, 1, s.nx) * scale
        if trial % 4 == 0:
            y[2] += 0.004
        dy = s.rhs(0.0, y)
        f = s._f.copy()                                   # forces on the foot points (world)
        q, v = y[:nq], y[nq:2 * nq]
        eta, etad = y[2 * nq:2 * nq + nm], y[2 * nq + nm:2 * nq + 2 * nm]
        a = dy[nq:2 * nq]
        # generalised contact force on the human: tau_c = M a + h - S^T tau
        tau = s.controller.torque(0.0, q, v, y[2 * nq + 2 * nm:2 * nq + 2 * nm + s.nc], h)[0]
        inv = pin.rnea(h.model, h.data, q, v, a).copy()
        tau_c = inv - tau
        # generalised contact force on the deck: Q_c = eta_dd + c eta_d + k eta (no shaker)
        Q_c = dy[2 * nq + nm:2 * nq + 2 * nm] + s.c_modal * etad + s.k_modal * eta
        # Newton's third law at the interface, resultant level
        assert np.abs(Q_c + s.S2.T @ f.ravel()).max() <= 1e-9 * max(np.abs(Q_c).max(), 1.0)
        for _ in range(3):
            dq, deta = rng.normal(0, 1, nq), rng.normal(0, 1, nm)
            # eps = 1e-5 balances truncation (~eps^2) against round-off
            # (~1e-16 * |x| / eps, with x ~ 7 m along the span)
            eps = 1e-5
            dP = (_points(s, q + eps * dq) - _points(s, q - eps * dq)) / (2 * eps)
            dU = (s.S2 @ deta).reshape(s.npts, 3)
            lhs = tau_c @ dq + Q_c @ deta
            rhs = np.sum(f * (dP - dU))
            # Normalise by the size of the terms that enter the identity.  lhs
            # and rhs are signed sums that can cancel for a random direction,
            # and tau_c is itself a difference of O(100 N m) quantities, so a
            # nearly open contact (forces of micro-newtons) must be judged
            # against that round-off floor, not against its own magnitude.
            norm = (np.sum(np.abs(f) * np.abs(dP - dU))
                    + np.sum(np.abs(inv * dq)) + np.sum(np.abs(tau * dq)))
            worst = max(worst, abs(lhs - rhs) / norm)
    assert worst < TOL["action_reaction_rel"]


@pytest.mark.parametrize("g", [9.81, 3.71, 1.62])
def test_centroidal_momentum_balance(g, modal_deck, rng):
    """m * a_com = sum of foot forces + m * g_vec, with a_com from the solved
    accelerations.  Joint torques are internal and must not appear."""
    sc = make_standing(g=g, deck=modal_deck)
    s, y0, h = sc.system, sc.y_eq, sc.human
    nq = s.nq
    m = h.total_mass
    for trial in range(10):
        y = y0.copy()
        y[:2 * nq] += rng.normal(0, 1, 2 * nq) * np.concatenate([np.full(36, 4e-3), np.full(36, 8e-2)])
        dy = s.rhs(0.0, y)
        ftot = s._f.sum(axis=0)
        pin.centerOfMass(h.model, h.data, y[:nq], y[nq:2 * nq], dy[nq:2 * nq])
        acom = h.data.acom[0]
        expect = ftot / m + np.array([0, 0, -g])
        assert np.abs(acom - expect).max() <= TOL["centroidal_rel"] * max(np.abs(expect).max(), g)
