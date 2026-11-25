"""Monolithic human-structure-contact system and its time integration.

State  y = [q (36), v (36), eta (nm), eta_dot (nm), x_c (nc), W (4)]
(x_c: internal states of the controller, e.g. the sway-feedback lag)

    M_h(q) vdot + h_h(q, v, g) = S^T tau + J_c(q)^T f_c            (human)
    eta'' + 2 zeta w eta' + w^2 eta = -S_c^T f_c + S_s^T F_s(t)    (deck, modal)

``f_c`` are the foot contact forces (acting on the foot); the SAME forces with
opposite sign load the deck through the surface interpolation matrices S_c
that are also used to read the deck motion at the contact points, so the two
subsystems exchange equal and opposite, work-conjugate forces.  Contact gaps
and slip velocities are measured against the DEFORMED, MOVING deck surface.

The whole right-hand side is evaluated at once for every Runge-Kutta stage
(no co-simulation lag).  ``W`` accumulates work integrals with the same
quadrature so that an energy balance can be audited:

    W[0] = int tau . v dt                 work of the joint torques
    W[1] = int sum f_c . (v_pt - v_surf)  work of the contact force pair on
                                          the two bodies (<= 0 net for a
                                          dissipative contact, up to stored
                                          elastic contact energy)
    W[2] = int sum 2 zeta w eta'^2 dt     energy dissipated by deck damping
    W[3] = int F_s . v_surface dt         work of the external deck forces

Balance:  d/dt [T_h + U_h + E_deck] = dW0 + dW1 - dW2 + dW3.

Gravity enters only through ``human.g`` (the gravity field of the multibody
model).  The deck matrices carry no gravity.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from ..contact.compliant import (ContactParams, contact_forces, elastic_energy,
                                 update_anchor)
from ..sensor.imu import imu_ideal
from .kernel import contact_deck_kernel, points_world

N_WORK = 4


class CoupledSystem:
    def __init__(self, human, contact: ContactParams, controller, deck=None,
                 shaker_xy=None, contact_enabled: bool = True, use_kernel: bool = True):
        self.h = human
        self.use_kernel = use_kernel
        self.cp = contact
        self.controller = controller
        self.deck = deck
        self.nq = human.model.nq
        self.nr = human.n_root
        self.iz = human.dof("root_z")
        self.nm = 0 if deck is None else deck.nm
        self.nc = int(getattr(controller, "n_state", 0))
        self.cj = [int(j) for j in human.contact_joint]
        self.cloc = np.array(human.contact_local, float)          # (nf, npf, 3)
        self.nf, self.npf = self.cloc.shape[:2]
        # skew matrices of the contact-point offsets: skew[i, k] @ w = loc x w
        sk = np.zeros((self.nf, self.npf, 3, 3))
        sk[..., 0, 1], sk[..., 0, 2] = -self.cloc[..., 2], self.cloc[..., 1]
        sk[..., 1, 0], sk[..., 1, 2] = self.cloc[..., 2], -self.cloc[..., 0]
        sk[..., 2, 0], sk[..., 2, 1] = -self.cloc[..., 1], self.cloc[..., 0]
        self._skew = sk
        # moment about the foot joint origin: sum_k loc_k x f_k = _mom[i] @ f.ravel()
        self._mom = np.concatenate([sk[:, k] for k in range(self.npf)], axis=2)
        self.npts = self.nf * self.npf
        self.contact_enabled = contact_enabled
        self.anchor = np.zeros((self.npts, 2))
        self.S2 = np.zeros((3 * self.npts, self.nm))               # surface modes at contacts
        self._S_ref = np.full((self.npts, 2), np.inf)
        self.k_modal = np.zeros(0)
        self.c_modal = np.zeros(0)
        if self.nm:
            self.k_modal = np.ascontiguousarray(deck.omega ** 2)
            self.c_modal = np.ascontiguousarray(2.0 * deck.zeta * deck.omega)
        # work arrays of the compiled path
        self._R = np.zeros((self.nf, 3, 3))
        self._p = np.zeros((self.nf, 3))
        self._vl = np.zeros((self.nf, 3))
        self._w = np.zeros((self.nf, 3))
        self._P = np.zeros((self.npts, 3))
        self._rel = np.zeros((self.npts, 3))
        self._relv = np.zeros((self.npts, 3))
        self._f = np.zeros((self.npts, 3))
        self._fn = np.zeros(self.npts)
        self._flin = np.zeros((self.nf, 3))
        self._fang = np.zeros((self.nf, 3))
        self._Qe0 = np.zeros(self.nm)
        self._par = np.array([contact.k_n, contact.n_exp, contact.c_n, contact.k_t,
                              contact.c_t, contact.mu], float)
        self.B = None
        self.shaker_xy = None
        if shaker_xy is not None and self.nm:
            self.shaker_xy = np.asarray(shaker_xy, float)
            Ss = deck.surface_modes(self.shaker_xy[:, 0], self.shaker_xy[:, 1])
            self.B = Ss.reshape(-1, self.nm).T.copy()              # (nm, 3*ns)
        # lateral position of the model's y = 0 plane on the deck (the planar
        # walker has no lateral coordinate; its walking line is placed here)
        self.y_shift = 0.0
        self.fext = pin.StdVec_Force()
        for _ in range(human.model.njoints):
            self.fext.append(pin.Force.Zero())
        self.last = {}
        self.n_slip = 0

    # ----------------------------------------------------------------- state
    @property
    def ny(self) -> int:
        return 2 * self.nq + 2 * self.nm + self.nc + N_WORK

    @property
    def nx(self) -> int:
        """Number of dynamic states (everything except the work integrals)."""
        return 2 * self.nq + 2 * self.nm + self.nc

    def pack(self, q, v, eta=None, etad=None):
        y = np.zeros(self.ny)
        y[:self.nq] = q
        y[self.nq:2 * self.nq] = v
        if self.nm:
            if eta is not None:
                y[2 * self.nq:2 * self.nq + self.nm] = eta
            if etad is not None:
                y[2 * self.nq + self.nm:2 * self.nq + 2 * self.nm] = etad
        return y

    def unpack(self, y):
        nq, nm = self.nq, self.nm
        return (y[:nq], y[nq:2 * nq], y[2 * nq:2 * nq + nm],
                y[2 * nq + nm:2 * nq + 2 * nm], y[-N_WORK:])

    # ------------------------------------------------------------- kinematics
    def _feet(self, q, v):
        """World positions / velocities of all contact points (FK must be current)."""
        d = self.h.data
        P = np.empty((self.npts, 3))
        V = np.empty((self.npts, 3))
        Rs = []
        for i, j in enumerate(self.cj):
            M = d.oMi[j]
            R = M.rotation
            mv = d.v[j]
            loc = self.cloc[i]
            sl = slice(self.npf * i, self.npf * (i + 1))
            P[sl] = loc @ R.T + M.translation
            # w x loc = -(loc x w)
            V[sl] = (mv.linear - self._skew[i] @ mv.angular) @ R.T
            Rs.append(R)
        return P, V, Rs

    def _refresh_S(self, xy, tol=2e-3, force=False, near=None):
        """Re-evaluate the surface mode matrices where contact locations moved.
        ``near`` restricts the update to points close to the surface (a foot in
        mid-swing does not need the deck shape under it)."""
        if not self.nm:
            return
        moved = np.linalg.norm(xy - self._S_ref, axis=1) > tol
        if near is not None:
            moved &= near
        if force:
            moved[:] = True
        if moved.any():
            S = self.deck.surface_modes(xy[moved, 0], xy[moved, 1] + self.y_shift)
            S3 = self.S2.reshape(self.npts, 3, self.nm)
            S3[moved] = S
            self._S_ref[moved] = xy[moved]

    def reset_anchors(self, y):
        """Put every anchor under its contact point (zero tangential force)."""
        q, v, eta, etad, _ = self.unpack(y)
        pin.forwardKinematics(self.h.model, self.h.data, q, v)
        P, V, _ = self._feet(q, v)
        self._refresh_S(P[:, :2], force=True)
        us = (self.S2 @ eta).reshape(self.npts, 3) if self.nm else 0.0
        self.anchor = (P - us)[:, :2].copy()

    # -------------------------------------------------------------------- rhs
    def rhs(self, t, y, F=None, first: bool = False):
        """Time derivative of the state.  ``first=True`` marks the first stage
        of a step, where the friction anchors are updated."""
        if self.use_kernel:
            return self._rhs_kernel(t, y, F, first)
        return self.rhs_reference(t, y, F, first)

    def _rhs_kernel(self, t, y, F=None, first: bool = False):
        """Compiled path; must stay equivalent to ``rhs_reference``."""
        h = self.h
        m, d = h.model, h.data
        nq, nm, nc = self.nq, self.nm, self.nc
        ie = 2 * nq
        q, v = y[:nq], y[nq:ie]
        eta, etad = y[ie:ie + nm], y[ie + nm:ie + 2 * nm]
        pin.forwardKinematics(m, d, q, v)
        R, p, vl, w = self._R, self._p, self._vl, self._w
        for i, j in enumerate(self.cj):
            M = d.oMi[j]
            R[i] = M.rotation
            p[i] = M.translation
            mv = d.v[j]
            vl[i] = mv.linear
            w[i] = mv.angular
        if first and self.contact_enabled and nm:
            points_world(R, p, self.cloc, self._P)
            self._refresh_S(self._P[:, :2], near=self._P[:, 2] < 0.03)
        Qe = self.B @ F if (F is not None and self.B is not None) else self._Qe0
        dy = np.empty_like(y)
        pc, pdm, pe, ns = contact_deck_kernel(
            R, p, vl, w, self.cloc, self.S2, eta, etad, self.anchor, self._par,
            self.k_modal, self.c_modal, Qe, first, self.contact_enabled,
            self._P, self._rel, self._relv, self._f, self._fn, self._flin, self._fang,
            dy[ie + nm:ie + 2 * nm])
        self.n_slip += ns
        for i, j in enumerate(self.cj):
            self.fext[j] = pin.Force(self._flin[i], self._fang[i])
        if nc:
            i0 = ie + 2 * nm
            tau, dxc = self.controller.torque(t, q, v, y[i0:i0 + nc], h, feet=(p, R, vl))
            dy[i0:i0 + nc] = dxc
        elif getattr(self.controller, "wants_system", False):
            tau = self.controller.torque_sys(t, q, v, self, first)
        else:
            tau = self.controller(t, q, v)
        a = pin.aba(m, d, q, v, tau, self.fext)
        dy[:nq] = v
        dy[nq:ie] = a
        dy[ie:ie + nm] = etad
        dy[-4] = tau @ v
        dy[-3] = pc
        dy[-2] = pdm
        dy[-1] = pe
        if first:
            self.last = dict(P=self._P.copy(), gap=self._rel[:, 2].copy(), f=self._f.copy(),
                             fn=self._fn.copy(), rel=self._rel.copy(), relv=self._relv.copy(),
                             s=self._rel[:, :2] - self.anchor, tau=tau, a=a)
            if not self.contact_enabled:
                self.last["fn"] = np.zeros(self.npts)
                self.last["f"] = np.zeros((self.npts, 3))
                self.last["gap"] = np.full(self.npts, np.inf)
        return dy

    def rhs_reference(self, t, y, F=None, first: bool = False):
        """Plain NumPy implementation (reference for the compiled path)."""
        h = self.h
        m, d = h.model, h.data
        nq, nm = self.nq, self.nm
        q, v = y[:nq], y[nq:2 * nq]
        eta = y[2 * nq:2 * nq + nm]
        etad = y[2 * nq + nm:2 * nq + 2 * nm]
        pin.forwardKinematics(m, d, q, v)
        dy = np.empty_like(y)
        p_contact = 0.0
        if self.contact_enabled:
            P, V, Rs = self._feet(q, v)
            if first:
                self._refresh_S(P[:, :2])
            if nm:
                us = (self.S2 @ eta).reshape(self.npts, 3)
                uds = (self.S2 @ etad).reshape(self.npts, 3)
                rel, relv = P - us, V - uds
            else:
                rel, relv = P, V
            gap, gap_rate = rel[:, 2], relv[:, 2]
            if first:
                fn0 = np.maximum(self.cp.k_n * np.maximum(-gap, 0.0) ** self.cp.n_exp
                                 * (1.0 - self.cp.c_n * gap_rate), 0.0)
                self.anchor, slip = update_anchor(self.cp, self.anchor, rel[:, :2], gap, fn0)
                self.n_slip += int(slip.sum())
            s = rel[:, :2] - self.anchor
            f, fn = contact_forces(self.cp, gap, gap_rate, s, relv[:, :2])
            for i, j in enumerate(self.cj):
                sl = slice(self.npf * i, self.npf * (i + 1))
                fl = f[sl] @ Rs[i]                       # rows = R^T f
                self.fext[j] = pin.Force(fl.sum(axis=0), self._mom[i] @ fl.ravel())
            p_contact = float(np.sum(f * relv))
            if first:
                self.last = dict(P=P, V=V, gap=gap.copy(), f=f, fn=fn, s=s.copy(),
                                 rel=rel, relv=relv)
        else:
            f = None
            for j in self.cj:
                self.fext[j] = pin.Force.Zero()
            if first:
                self.last = dict(P=np.zeros((self.npts, 3)), gap=np.full(self.npts, np.inf),
                                 f=np.zeros((self.npts, 3)), fn=np.zeros(self.npts))
        if self.nc:
            i0 = 2 * nq + 2 * nm
            tau, dxc = self.controller.torque(t, q, v, y[i0:i0 + self.nc], h)
            dy[i0:i0 + self.nc] = dxc
        else:
            tau = self.controller(t, q, v)
        a = pin.aba(m, d, q, v, tau, self.fext)
        dy[:nq] = v
        dy[nq:2 * nq] = a
        p_ext = 0.0
        p_damp = 0.0
        if nm:
            Q = -self.S2.T @ f.ravel() if f is not None else np.zeros(nm)
            if F is not None and self.B is not None:
                Qe = self.B @ F
                Q = Q + Qe
                p_ext = float(Qe @ etad)
            dy[2 * nq:2 * nq + nm] = etad
            dy[2 * nq + nm:2 * nq + 2 * nm] = Q - self.c_modal * etad - self.k_modal * eta
            p_damp = float(self.c_modal @ (etad * etad))
        dy[-4] = float(tau @ v)
        dy[-3] = p_contact
        dy[-2] = p_damp
        dy[-1] = p_ext
        if first:
            self.last["tau"] = tau
            self.last["a"] = a
        return dy

    def step(self, t, y, dt, F0=None, Fh=None, F1=None):
        """One classical RK4 step.  Returns (y_next, k1)."""
        k1 = self.rhs(t, y, F0, first=True)
        k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, Fh)
        k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, Fh)
        k4 = self.rhs(t + dt, y + dt * k3, F1)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), k1

    # ---------------------------------------------------------------- energy
    def energy(self, y) -> dict:
        q, v, eta, etad, W = self.unpack(y)
        T = self.h.kinetic_energy(q, v)
        U = self.h.potential_energy(q)
        Ed = 0.5 * float(etad @ etad + self.k_modal @ (eta * eta)) if self.nm else 0.0
        Ec = 0.0
        if self.contact_enabled:
            pin.forwardKinematics(self.h.model, self.h.data, q, v)
            P, V, _ = self._feet(q, v)
            us = (self.S2 @ eta).reshape(self.npts, 3) if self.nm else 0.0
            uds = (self.S2 @ etad).reshape(self.npts, 3) if self.nm else 0.0
            rel, relv = P - us, V - uds
            s = rel[:, :2] - self.anchor
            _, fn = contact_forces(self.cp, rel[:, 2], relv[:, 2], s, relv[:, :2])
            Ec = elastic_energy(self.cp, rel[:, 2], s, fn)
        mech = T + U + Ed
        return dict(T_h=T, U_h=U, E_deck=Ed, E_contact=Ec, mech=mech,
                    W_act=float(W[0]), W_contact=float(W[1]), D_deck=float(W[2]),
                    W_ext=float(W[3]),
                    # zero if forces are applied consistently and integration is accurate
                    balance=mech - (W[0] + W[1] - W[2] + W[3]))

    # ------------------------------------------------------------ simulation
    def deck_lead_in(self, forcing, T_lead, dt):
        """The deck alone under the shakers, from rest, over [-T_lead, 0).

        the earlier study (Section 2.5) discards a deck-alone lead-in under the same
        continuous shaker realization, so that the walker meets a deck that is
        already in its stationary random vibration.  The walker is off the
        deck during this interval, so the deck's modal equations are
        integrated alone, with the RK4 scheme and step of ``simulate`` and the
        force sampled at the same half steps.  Returns (eta, etad) at t = 0.
        """
        eta, etad = np.zeros(self.nm), np.zeros(self.nm)
        if not self.nm or forcing is None or self.B is None or T_lead <= 0:
            return eta, etad
        n = int(round(T_lead / dt))
        Q = self.B @ forcing.sample(0.5 * dt, 2 * n + 1, -n * dt)      # (nm, 2n+1) modal forces
        k, c = self.k_modal, self.c_modal
        for i in range(n):
            q0, qh, q1 = Q[:, 2 * i], Q[:, 2 * i + 1], Q[:, 2 * i + 2]
            a1 = q0 - c * etad - k * eta
            e2, v2 = eta + 0.5 * dt * etad, etad + 0.5 * dt * a1
            a2 = qh - c * v2 - k * e2
            e3, v3 = eta + 0.5 * dt * v2, etad + 0.5 * dt * a2
            a3 = qh - c * v3 - k * e3
            e4, v4 = eta + dt * v3, etad + dt * a3
            a4 = q1 - c * v4 - k * e4
            eta = eta + dt / 6.0 * (etad + 2 * v2 + 2 * v3 + v4)
            etad = etad + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        return eta, etad

    def simulate(self, y0, T, dt, out_dt=None, forcing=None, t0=0.0,
                 surface_out_xy=None, fall_height=None, record_q=False,
                 stop_on_fall=True):
        """Integrate for ``T`` seconds with fixed step ``dt``.

        Outputs are sampled every ``out_dt`` (a multiple of dt) at the START
        of a step, from the same right-hand-side evaluation that advances the
        solution.  ``forcing`` is a Multisine with 3 channels per shaker point
        (x, y, z), or None.
        """
        n = int(round(T / dt))
        every = 1 if out_dt is None else max(1, int(round(out_dt / dt)))
        nout = n // every + 1
        Fh = None
        if forcing is not None and self.B is not None:
            Fh = forcing.sample(0.5 * dt, 2 * n + 1, t0).T.copy()      # (2n+1, nchan)
        S_out = None
        if surface_out_xy is not None and self.nm:
            xy = np.asarray(surface_out_xy, float)
            S_out = self.deck.surface_modes(xy[:, 0], xy[:, 1]).reshape(-1, self.nm)
        names = list(self.h.sensor_frame)
        out = dict(t=np.zeros(nout), sensors=names,
                   f=np.zeros((nout, len(names), 3)), w=np.zeros((nout, len(names), 3)),
                   Rz=np.zeros((nout, len(names), 3)),
                   fn=np.zeros((nout, self.npts)), fc=np.zeros((nout, self.nf, 3)),
                   cop=np.zeros((nout, self.nf, 2)), com=np.zeros((nout, 3)),
                   root=np.zeros((nout, self.nr)), tau=np.zeros((nout, self.nq - self.nr)),
                   energy=np.zeros((nout, 9)), min_gap=np.zeros(nout),
                   energy_cols=["T_h", "U_h", "E_deck", "E_contact", "W_act",
                                "W_contact", "D_deck", "W_ext", "balance"])
        if self.nm:
            out["eta"] = np.zeros((nout, self.nm))
            out["etadd"] = np.zeros((nout, self.nm))
            out["a_foot"] = np.zeros((nout, self.nf, 3))    # deck surface acceleration under each foot
            if S_out is not None:
                out["a_surf"] = np.zeros((nout, S_out.shape[0] // 3, 3))
        if record_q:
            out["q"] = np.zeros((nout, self.nq))
            out["v"] = np.zeros((nout, self.nq))
        y = np.array(y0, float)
        t = t0
        fell = False
        z0 = y[self.iz]
        io = 0
        nq, nm = self.nq, self.nm
        for i in range(n + 1):
            F0 = None if Fh is None else Fh[2 * i]
            if i % every == 0 or i == n:
                k1 = self.rhs(t, y, F0, first=True)
                if i % every == 0 and io < nout:
                    self._record(out, io, t, y, k1, S_out)
                    io += 1
                if i == n:
                    break
                k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, None if Fh is None else Fh[2 * i + 1])
                k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, None if Fh is None else Fh[2 * i + 1])
                k4 = self.rhs(t + dt, y + dt * k3, None if Fh is None else Fh[2 * i + 2])
                y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                y, _ = self.step(t, y, dt, F0, Fh[2 * i + 1] if Fh is not None else None,
                                 Fh[2 * i + 2] if Fh is not None else None)
            t = t0 + (i + 1) * dt
            if not np.all(np.isfinite(y)):
                fell = True
                out["failure"] = f"non-finite state at t={t:.4f}"
                break
            if fall_height is not None and y[self.iz] < fall_height * z0:
                fell = True
                out["failure"] = f"root height below {fall_height:.2f} of initial at t={t:.4f}"
                if stop_on_fall:
                    break
        for k in list(out):
            if isinstance(out[k], np.ndarray) and out[k].shape[:1] == (nout,):
                out[k] = out[k][:io]
        out.update(y_end=y, t_end=t, fell=fell, dt=dt, n_slip=self.n_slip,
                   out_dt=every * dt)
        return out

    def _record(self, out, io, t, y, k1, S_out):
        nq, nm = self.nq, self.nm
        q, v = y[:nq], y[nq:2 * nq]
        a = k1[nq:2 * nq]
        out["t"][io] = t
        L = self.last
        if self.contact_enabled:
            out["fn"][io] = L["fn"]
            out["min_gap"][io] = L["gap"].min()
            for i in range(self.nf):
                sl = slice(self.npf * i, self.npf * (i + 1))
                out["fc"][io, i] = L["f"][sl].sum(axis=0)
                tot = L["fn"][sl].sum()
                if tot > 0:
                    out["cop"][io, i] = (L["fn"][sl][:, None] * L["P"][sl][:, :2]).sum(axis=0) / tot
                else:
                    out["cop"][io, i] = np.nan
        out["tau"][io] = L["tau"][self.nr:]
        imu = imu_ideal(self.h, q, v, a)
        for k, name in enumerate(out["sensors"]):
            out["f"][io, k] = imu[name]["f"]
            out["w"][io, k] = imu[name]["w"]
            out["Rz"][io, k] = imu[name]["R"][2]        # world-z expressed in sensor axes
        c, _ = self.h.com(q, v)
        out["com"][io] = c
        out["root"][io] = q[:self.nr]
        e = self.energy(y)
        out["energy"][io] = [e[k] for k in out["energy_cols"]]
        if nm:
            out["eta"][io] = y[2 * nq:2 * nq + nm]
            etadd = k1[2 * nq + nm:2 * nq + 2 * nm]
            out["etadd"][io] = etadd
            S3 = self.S2.reshape(self.npts, 3, nm)
            for i in range(self.nf):
                out["a_foot"][io, i] = S3[self.npf * i:self.npf * (i + 1)].mean(axis=0) @ etadd
            if S_out is not None:
                out["a_surf"][io] = (S_out @ etadd).reshape(-1, 3)
        if "q" in out:
            out["q"][io], out["v"][io] = q, v

    # ----------------------------------------------------------- linearisation
    def linearize(self, y_eq, eps: float = 1e-6):
        """Finite-difference linearisation about a (static) equilibrium.

        Returns A (nx, nx) for x = [q, v, eta, eta_dot, x_c], the input matrix
        B_F for the shaker forces and the residual f(x_eq).  Friction anchors
        are held fixed (sticking contact).
        """
        nx = self.nx
        y0 = np.array(y_eq, float)
        f0 = self.rhs(0.0, y0)[:nx]
        A = np.zeros((nx, nx))
        for i in range(nx):
            yp, ym = y0.copy(), y0.copy()
            yp[i] += eps
            ym[i] -= eps
            A[:, i] = (self.rhs(0.0, yp)[:nx] - self.rhs(0.0, ym)[:nx]) / (2 * eps)
        nF = 0 if self.B is None else self.B.shape[1]
        BF = np.zeros((nx, nF))
        if nF:
            BF[2 * self.nq + self.nm:2 * self.nq + 2 * self.nm, :] = self.B
        return A, BF, f0

    def force_jacobian(self, y_eq, eps: float = 1e-6):
        """d(total foot force, world axes)/d(x) at an equilibrium, shape (3, nx)."""
        nx = self.nx
        y0 = np.array(y_eq, float)

        def g(y):
            self.rhs(0.0, y)
            return self._f.sum(axis=0).copy()

        C = np.zeros((3, nx))
        for i in range(nx):
            yp, ym = y0.copy(), y0.copy()
            yp[i] += eps
            ym[i] -= eps
            C[:, i] = (g(yp) - g(ym)) / (2 * eps)
        return C

    def sensor_jacobians(self, y_eq, eps: float = 1e-6, g_out=None):
        """d(specific force, angular rate)/d(x) and d/d(F) at an equilibrium.

        Rows are ordered sensor by sensor: f_x, f_y, f_z, w_x, w_y, w_z.
        """
        nx = self.nx
        nq = self.nq
        nF = 0 if self.B is None else self.B.shape[1]
        names = list(self.h.sensor_frame)

        def g(y, F=None):
            k = self.rhs(0.0, y, F)
            imu = imu_ideal(self.h, y[:nq], y[nq:2 * nq], k[nq:2 * nq], g_out=g_out)
            return np.concatenate([np.concatenate([imu[n]["f"], imu[n]["w"]]) for n in names])

        y0 = np.array(y_eq, float)
        F0 = np.zeros(nF) if nF else None
        g0 = g(y0, F0)
        C = np.zeros((g0.size, nx))
        for i in range(nx):
            yp, ym = y0.copy(), y0.copy()
            yp[i] += eps
            ym[i] -= eps
            C[:, i] = (g(yp, F0) - g(ym, F0)) / (2 * eps)
        D = np.zeros((g0.size, nF))
        for i in range(nF):
            Fp, Fm = F0.copy(), F0.copy()
            Fp[i] += 1.0
            Fm[i] -= 1.0
            D[:, i] = (g(y0, Fp) - g(y0, Fm)) / 2.0
        return C, D, g0, names


def rk4_max_stable_dt(A) -> float:
    """Largest fixed step for which classical RK4 is linearly stable for the
    system matrix ``A`` (bisection on max |R(lambda*dt)| <= 1)."""
    lam = np.linalg.eigvals(A)

    def ok(dt):
        z = lam * dt
        return np.abs(1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24).max() <= 1.0 + 1e-9

    lo, hi = 1e-7, 1e-1
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(lo)
