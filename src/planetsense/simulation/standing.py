"""Standing scenarios: gravity-specific static equilibrium and scenario assembly.

For every gravity (and, later, suit condition) the static configuration is
re-solved rather than copied from Earth:

unknowns   root height, root pitch, root roll, deck modal coordinates
equations  no net vertical force, no net pitch / roll moment on the
           (unactuated) root, and static equilibrium of the deck under the
           foot forces.

The joint feed-forward torques then follow from inverse statics.  The three
remaining root equations (horizontal forces, yaw moment) are reported as a
residual; they vanish for a symmetric posture because the tangential contact
springs are unloaded at the solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from ..contact.compliant import ContactParams, contact_forces, static_penetration
from ..coupling.system import CoupledSystem
from ..excitation.multisine import shaker_layout
from ..human.control import LegSpring, PostureController, posture_gains, standing_posture
from ..human.model import build_human

G_EARTH, G_MARS, G_MOON = 9.81, 3.71, 1.62     # nominal values


@dataclass
class StandingScenario:
    system: CoupledSystem
    y_eq: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def human(self):
        return self.system.h


def _static_residual(sys: CoupledSystem, q, eta):
    """Generalised force that would be needed to hold (q, eta) statically."""
    h = sys.h
    m, d = h.model, h.data
    zero = np.zeros(sys.nq)
    pin.forwardKinematics(m, d, q, zero)
    P, V, Rs = sys._feet(q, zero)
    sys._refresh_S(P[:, :2], force=True)
    us = (sys.S2 @ eta).reshape(sys.npts, 3) if sys.nm else 0.0
    rel = P - us
    f, fn = contact_forces(sys.cp, rel[:, 2], np.zeros(sys.npts),
                           np.zeros((sys.npts, 2)), np.zeros((sys.npts, 2)))
    for i, j in enumerate(sys.cj):
        sl = slice(sys.npf * i, sys.npf * (i + 1))
        fl = f[sl] @ Rs[i]
        sys.fext[j] = pin.Force(fl.sum(axis=0), sys._mom[i] @ fl.ravel())
    gen = pin.rnea(m, d, q, zero, zero, sys.fext)            # g(q) - J^T f
    r_deck = (sys.k_modal * eta + sys.S2.T @ f.ravel()) if sys.nm else np.zeros(0)
    return gen, r_deck, f, fn, P


def solve_equilibrium(sys: CoupledSystem, q_posture, x_com: float, y_com: float,
                      yaw: float = 0.0, tol: float = 1e-10):
    """Solve the standing equilibrium.  ``q_posture`` supplies the joint angles
    and an initial root height; the body is placed so that its centre of mass
    is above (x_com, y_com).  Passive leg springs, if present, add their static
    compressions to the unknowns."""
    h = sys.h
    ix, iy, iz = h.dof("root_x"), h.dof("root_y"), h.dof("root_z")
    iyaw, ipit, irol = h.dof("root_rz"), h.dof("root_ry"), h.dof("root_rx")
    ctrl = sys.controller
    legs = list(getattr(ctrl, "leg_idx", ()))
    q = np.array(q_posture, float)
    q[iyaw] = yaw
    c = h.com(q)
    q[ix] += x_com - c[0]
    q[iy] += y_com - c[1]
    W = h.total_mass * h.g
    d0 = static_penetration(sys.cp, W / sys.npts)
    dl = ctrl.leg.static_compression(W / max(len(legs), 1)) if legs else 0.0
    z0 = q[iz] - d0 - dl
    sc_f, sc_m = W, W * 0.1
    k_ref = sys.k_modal if sys.nm else None
    nl = len(legs)

    def unpack(u):
        qq = q.copy()
        qq[iz], qq[ipit], qq[irol] = z0 + u[0] * (d0 + dl), u[1] * 1e-3, u[2] * 1e-3
        if nl:
            qq[legs] = dl * (1.0 + u[3:3 + nl])
        eta = u[3 + nl:] * (W / k_ref) if sys.nm else np.zeros(0)
        return qq, eta

    def res(u):
        qq, eta = unpack(u)
        gen, r_deck, *_ = _static_residual(sys, qq, eta)
        r = [gen[iz] / sc_f, gen[ipit] / sc_m, gen[irol] / sc_m]
        if nl:      # the passive spring must supply the static axial load of each leg
            r += list((gen[legs] + ctrl.leg.force(qq[legs], np.zeros(nl))) / sc_f)
        return np.concatenate([r, r_deck / W]) if sys.nm else np.array(r)

    sol = least_squares(res, np.zeros(3 + nl + sys.nm), xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        x_scale="jac", max_nfev=300)
    q_eq, eta_eq = unpack(sol.x)
    gen, r_deck, f, fn, P = _static_residual(sys, q_eq, eta_eq)
    y_eq = sys.pack(q_eq, np.zeros(sys.nq), eta_eq)
    sys.reset_anchors(y_eq)
    tau_ff = gen[h.n_root:].copy()
    if nl:
        tau_ff[[i - h.n_root for i in legs]] = 0.0        # carried by the passive springs
    rr = gen[:h.n_root].copy()
    info = dict(residual_root=rr, residual_norm=float(np.abs(res(sol.x)).max()),
                tau_ff=tau_ff, fn=fn.copy(), P=P.copy(), weight=W,
                sum_fn=float(fn.sum()), nfev=int(sol.nfev),
                penetration=np.maximum(-(P[:, 2] - ((sys.S2 @ eta_eq).reshape(sys.npts, 3)[:, 2]
                                                    if sys.nm else 0.0)), 0.0),
                com=h.com(q_eq), eta=eta_eq.copy(),
                leg_compression=q_eq[legs].copy() if nl else np.zeros(0))
    tot = fn.sum()
    info["cop"] = (fn[:, None] * P[:, :2]).sum(axis=0) / tot
    return y_eq, info


def make_standing(g: float = G_EARTH, deck=None, contact: ContactParams | None = None,
                  x_com: float = 7.0, y_com: float = 0.575, gain_policy: str = "gravity_scaled",
                  kappa: float = 3.0, zeta: float = 0.5, kappa_ankle: float | None = 0.9,
                  zeta_ankle: float | None = 0.05,
                  kappa_sway: float = 1.5, T_d: float = 0.30, tau_f: float = 0.15,
                  human_kwargs: dict | None = None,
                  posture_kwargs: dict | None = None, shakers: bool = True,
                  leg: LegSpring | None = None) -> StandingScenario:
    """Assemble a standing person on a rigid floor (``deck=None``) or on a
    modal deck, solved to static equilibrium under gravity ``g``.

    ``gain_policy``: "gravity_scaled" (joint impedance follows the local
    weight) or "earth_fixed" (Earth impedance kept on any planet).  Both are
    provisional bounding assumptions, see ``planetsense.human.control``.
    """
    contact = contact or ContactParams()
    hk = dict(human_kwargs or {})
    if leg is not None:
        hk["leg_axial"] = True
    human = build_human(g=g, **hk)
    q_ref = standing_posture(human, **(posture_kwargs or {}))
    g_gain = g if gain_policy == "gravity_scaled" else G_EARTH
    if gain_policy not in ("gravity_scaled", "earth_fixed"):
        raise ValueError(gain_policy)
    kp, kd, table = posture_gains(human, q_ref, g_gain, kappa=kappa, zeta=zeta,
                                  kappa_ankle=kappa_ankle, zeta_ankle=zeta_ankle)
    ctrl = PostureController(
        q_ref=q_ref.copy(), kp=kp, kd=kd, tau_ff=np.zeros(human.nv - human.n_root),
        meta=dict(policy=gain_policy, g_gain=g_gain, kappa=kappa, zeta=zeta,
                  kappa_ankle=kappa_ankle, zeta_ankle=zeta_ankle,
                  kappa_sway=kappa_sway, T_d=T_d, tau_f=tau_f, table=table),
        K_s=kappa_sway * human.total_mass * g_gain, T_d=T_d, tau_f=tau_f,
        ankle_ry=(human.dof("ankle_L_ry"), human.dof("ankle_R_ry")),
        ankle_rx=(human.dof("ankle_L_rx"), human.dof("ankle_R_rx")),
        foot_joints=tuple(int(j) for j in human.contact_joint),
        n_root=human.n_root,
        leg_idx=((human.dof("ankle_L_z"), human.dof("ankle_R_z")) if leg is not None else ()),
        leg=leg)
    sxy = None
    if deck is not None and shakers:
        sxy = shaker_layout(deck.params.L, deck.params.W)
    sys = CoupledSystem(human, contact, ctrl, deck=deck, shaker_xy=sxy)
    y_eq, info = solve_equilibrium(sys, q_ref, x_com, y_com)
    ctrl.q_ref = y_eq[:sys.nq].copy()
    ctrl.tau_ff = info["tau_ff"].copy()
    if ctrl.n_state:
        pin.forwardKinematics(human.model, human.data, y_eq[:sys.nq], np.zeros(sys.nq))
        ctrl.c_ref = ctrl.sway(human, y_eq[:sys.nq], np.zeros(sys.nq))[0].copy()
    info.update(g=g, gain_policy=gain_policy, contact=contact.asdict(),
                gain_table=table, leg=None if leg is None else leg.__dict__.copy())
    return StandingScenario(system=sys, y_eq=y_eq, info=info)
