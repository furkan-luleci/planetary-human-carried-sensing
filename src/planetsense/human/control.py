"""Joint-space posture control for standing.

    tau_j = tau_ff,j - Kp_j (q_j - q_ref,j) - Kd_j v_j        (30 actuated DOFs)

The six root coordinates are never actuated: the body is held up only by
contact forces.  ``tau_ff`` is the static joint torque of the gravity- and
suit-specific equilibrium (recomputed for every gravity).

GAIN RULE (provisional assumption for the pilot, not a measured human
property).  For the joints that carry the body like an inverted pendulum
(ankle, knee, hip, lumbar, neck) the proportional gain is a multiple ``kappa``
of the gravitational toppling stiffness of everything above the joint,

    G_j = sum_{i above j} m_i g (z_i - z_j),        Kp_j = kappa * G_j / n_legs,

and the derivative gain is  Kd_j = 2 zeta sqrt(kappa G_j I_j) / n_legs,  with
I_j the moment of inertia of the supported mass about the joint (a damping
ratio ``zeta`` relative to the joint's own stiffness scale).  Joint impedance
alone needs kappa of about 6 to keep the multi-link body upright in this
model; with physiological values the body is stabilised by the low-bandwidth
sway feedback in ``PostureController`` instead.  Hanging joints
(shoulder, elbow, wrist) get a small constant stiffness ``k_hang`` and damping
relative to their pendulum frequency.

Two gravity policies bound the unknown dependence of human joint impedance on
gravity:

* ``g_gain = local g``   ("gravity-scaled"): impedance follows the load.
* ``g_gain = 9.81``      ("Earth-fixed"):    impedance unchanged off Earth.

Neither is a prediction of what people do on the Moon or Mars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from scipy.optimize import brentq

_LEG_ABOVE = {
    "ankle": lambda n: not n.startswith("foot"),
    "knee": lambda n: not (n.startswith("foot") or n.startswith("shank")),
    "hip": lambda n: not (n.startswith("foot") or n.startswith("shank") or n.startswith("thigh")),
}
_TRUNK_ABOVE = {
    "lumbar": lambda n: n in ("torso", "head") or "arm" in n or n.startswith("hand"),
    "neck": lambda n: n == "head",
}
_ARM_BELOW = {
    "shoulder": lambda n, s: n in (f"upper_arm_{s}", f"forearm_{s}", f"hand_{s}"),
    "elbow": lambda n, s: n in (f"forearm_{s}", f"hand_{s}"),
    "wrist": lambda n, s: n == f"hand_{s}",
}


@dataclass(frozen=True)
class LegSpring:
    """Passive axial compliance of one leg (the foot slides along the shank).

    d = leg shortening (m), positive in compression.

        F(d, d') = k d^n + damping            for d >= 0
        F(d, d') = k_ext d + c_ext d'         for d <  0  (holds the hanging foot)

    damping = c d'            ("constant", Kelvin-Voigt)
            = k d^n c d'      ("proportional", Hunt-Crossley form, c in s/m)

    n = 1 gives a linear leg whose stiffness cannot depend on gravity; n > 1
    stiffens with the load carried, so its small-vibration stiffness follows
    the local weight.  k and c are CALIBRATED to published summary values of
    the standing body's vertical apparent mass (docs/assumptions_and_sources.md);
    the exponent and the damping form are assumptions carried as uncertainty.
    """
    k: float = 6.4e4
    n: float = 1.0
    c: float = 750.0
    damping: str = "constant"
    k_ext: float = 2.0e4
    c_ext: float = 200.0

    def force(self, d, dd):
        dp = np.maximum(d, 0.0)
        el = self.k * dp ** self.n
        if self.damping == "constant":
            comp = el + self.c * dd
        else:
            comp = el * (1.0 + self.c * dd)
        return np.where(d >= 0.0, comp, self.k_ext * d + self.c_ext * dd)

    def energy(self, d):
        dp = np.maximum(d, 0.0)
        return float(np.sum(np.where(d >= 0.0, self.k * dp ** (self.n + 1.0) / (self.n + 1.0),
                                     0.5 * self.k_ext * d * d)))

    def static_compression(self, load):
        return float((load / self.k) ** (1.0 / self.n))

    def tangent(self, load):
        return float(self.n * load / self.static_compression(load))


@dataclass
class PostureController:
    """Joint impedance + feed-forward + low-bandwidth sway feedback.

    Sway feedback ("ankle strategy").  Let c be the horizontal position of the
    whole-body centre of mass relative to the midpoint of the two ankle joint
    centres, and e = c - c_ref.  A first-order lag with time constant
    ``tau_f`` (a crude stand-in for neural delay) filters the lead signal
    e + T_d * de/dt:

        tau_f * dx_f/dt = (e + T_d * de/dt) - x_f

    and the filtered signal drives both ankles,

        d tau_ankle_ry = +0.5 * K_s * x_f[0]        (plantar-flexion when the
                                                     centre of mass is forward)
        d tau_ankle_rx = -0.5 * K_s * x_f[1]

    with K_s = kappa_s * m * g_gain (N m per metre of sway, both ankles).  The
    centre of pressure then moves kappa_s times as far as the centre of mass
    at low frequency, which is stabilising for kappa_s > 1.  The feedback is
    translation invariant: it uses no absolute world coordinate.
    """
    q_ref: np.ndarray        # (36,) reference configuration (joint part is used)
    kp: np.ndarray           # (30,)
    kd: np.ndarray           # (30,)
    tau_ff: np.ndarray       # (30,)
    meta: dict
    K_s: float = 0.0         # N m / m, sway feedback gain (0 disables it)
    T_d: float = 0.30        # s, derivative (lead) time
    tau_f: float = 0.15      # s, lag time constant
    c_ref: np.ndarray | None = None      # (2,) reference sway offset
    ankle_ry: tuple = ()     # indices (into v) of the two ankle RY DOFs
    ankle_rx: tuple = ()
    foot_joints: tuple = ()  # pinocchio joint ids of the feet
    n_root: int = 6          # unactuated root coordinates
    leg_idx: tuple = ()      # indices (into v) of the leg axial coordinates
    leg: LegSpring | None = None

    def _joint_part(self, q, v):
        nr = self.n_root
        tau = np.zeros(q.size)
        tau[nr:] = self.tau_ff - self.kp * (q[nr:] - self.q_ref[nr:]) - self.kd * v[nr:]
        if self.leg_idx:
            i = list(self.leg_idx)
            tau[i] -= self.leg.force(q[i], v[i])       # passive, resists shortening
        return tau

    def elastic_energy(self, q) -> float:
        """Energy stored in the passive leg springs (J)."""
        if not self.leg_idx:
            return 0.0
        return self.leg.energy(q[list(self.leg_idx)])

    @property
    def n_state(self) -> int:
        return 2 if self.K_s > 0.0 else 0

    def sway(self, human, q, v, feet=None):
        """Centre of mass relative to the ankle midpoint, and its rate (2,), (2,).
        Forward kinematics with velocities must be current.  ``feet`` may pass
        the already extracted (p, R, v_local) arrays of the foot joints."""
        m, d = human.model, human.data
        pin.centerOfMass(m, d, q, v)
        c, cd = d.com[0], d.vcom[0]
        if feet is None:
            pa = np.zeros(2)
            va = np.zeros(2)
            for j in self.foot_joints:
                M = d.oMi[j]
                pa += M.translation[:2]
                va += (M.rotation @ d.v[j].linear)[:2]
            n = len(self.foot_joints)
            return c[:2] - pa / n, cd[:2] - va / n
        p, R, vl = feet
        n = p.shape[0]
        vw = np.einsum("iab,ib->a", R, vl)
        return (c[:2] - p[:, :2].sum(axis=0) / n), (cd[:2] - vw[:2] / n)

    def torque(self, t, q, v, xc, human, feet=None):
        """Return (tau (nv,), d xc/dt)."""
        tau = self._joint_part(q, v)
        if self.K_s <= 0.0:
            return tau, xc
        e, ed = self.sway(human, q, v, feet)
        dxc = ((e - self.c_ref) + self.T_d * ed - xc) / self.tau_f
        h = 0.5 * self.K_s
        for i in self.ankle_ry:
            tau[i] += h * xc[0]
        for i in self.ankle_rx:
            tau[i] -= h * xc[1]
        return tau, dxc

    def __call__(self, t, q, v):
        """Joint-impedance part only (kept for stateless use and tests)."""
        return self._joint_part(q, v)


def _segment_world(human, q):
    m, d = human.model, human.data
    pin.forwardKinematics(m, d, q)
    out = {}
    for name, j in human.joint_id.items():
        I = m.inertias[j]
        out[name] = dict(m=I.mass, c=d.oMi[j].act(I.lever), j=d.oMi[j].translation.copy(),
                         Iyy=(d.oMi[j].rotation @ I.inertia @ d.oMi[j].rotation.T)[1, 1])
    return out


def _g_and_i(sw, names, pivot, g):
    G = sum(sw[n]["m"] * g * (sw[n]["c"][2] - pivot[2]) for n in names)
    I = sum(sw[n]["m"] * ((sw[n]["c"][0] - pivot[0]) ** 2 + (sw[n]["c"][2] - pivot[2]) ** 2)
            + sw[n]["Iyy"] for n in names)
    return float(G), float(I)


def posture_gains(human, q_ref, g_gain: float, kappa: float = 3.0, zeta: float = 0.5,
                  k_hang: float = 5.0, zeta_hang: float = 0.3, axial_ratio: float = 0.3,
                  kappa_ankle: float | None = None, zeta_ankle: float | None = None):
    """Return (kp, kd, table) for the 30 actuated DOFs; see the module docstring.

    ``kappa_ankle`` / ``zeta_ankle`` override ``kappa`` / ``zeta`` for the
    ankles (intrinsic ankle stiffness is reported to be below the toppling
    stiffness in quiet standing and intrinsic ankle viscosity is small;
    stability and sway damping then come from the sway feedback)."""
    sw = _segment_world(human, q_ref)
    names = list(human.segments)
    kp = np.zeros(human.nv)
    kd = np.zeros(human.nv)
    table = {}

    def assign(joint, kp_j, kd_j):
        sl = human.vslice[joint]
        for i in range(sl.start, sl.stop):
            ax = human.dof_names[i].rsplit("_", 1)[1]
            if ax in ("x", "y", "z"):
                continue                     # prismatic coordinates carry no joint impedance
            f = axial_ratio if ax == "rz" else 1.0
            kp[i], kd[i] = f * kp_j, f * kd_j

    for side in ("L", "R"):
        for jn, seg_name in (("ankle", "foot"), ("knee", "shank"), ("hip", "thigh")):
            above = [n for n in names if _LEG_ABOVE[jn](n)]
            G, I = _g_and_i(sw, above, sw[f"{seg_name}_{side}"]["j"], g_gain)
            kap = kappa_ankle if (jn == "ankle" and kappa_ankle is not None) else kappa
            zt = zeta_ankle if (jn == "ankle" and zeta_ankle is not None) else zeta
            kpj = kap * G / 2.0
            # damping relative to the joint's own stiffness scale (kap*G), so
            # that it stays defined when kap <= 1
            kdj = 2.0 * zt * np.sqrt(kap * G * I) / 2.0
            assign(f"{jn}_{side}", kpj, kdj)
            table[f"{jn}_{side}"] = dict(G=G, I=I, kp=kpj, kd=kdj)
        for jn, seg_name in (("shoulder", "upper_arm"), ("elbow", "forearm"), ("wrist", "hand")):
            below = [n for n in names if _ARM_BELOW[jn](n, side)]
            G, I = _g_and_i(sw, below, sw[f"{seg_name}_{side}"]["j"], g_gain)   # G < 0
            k_eff = k_hang - G
            kdj = 2.0 * zeta_hang * np.sqrt(max(k_eff, 0.0) * I)
            assign(f"{jn}_{side}", k_hang, kdj)
            table[f"{jn}_{side}"] = dict(G=G, I=I, kp=k_hang, kd=kdj)
    for jn, seg_name in (("lumbar", "torso"), ("neck", "head")):
        above = [n for n in names if _TRUNK_ABOVE[jn](n)]
        G, I = _g_and_i(sw, above, sw[seg_name]["j"], g_gain)
        kpj = kappa * G
        kdj = 2.0 * zeta * np.sqrt(kappa * G * I)
        assign(jn, kpj, kdj)
        table[jn] = dict(G=G, I=I, kp=kpj, kd=kdj)
    return kp[human.n_root:], kd[human.n_root:], table


def standing_posture(human, knee_flex_deg: float = 5.0, elbow_flex_deg: float = 10.0,
                     com_ahead_of_ankle: float = 0.048) -> np.ndarray:
    """Symmetric standing reference posture with the pelvis upright.

    Knee flexion is prescribed; hip flexion ``a`` is solved so that the centre
    of mass lies ``com_ahead_of_ankle`` metres anterior to the ankle joints,
    with the ankle angle chosen to keep the feet flat (hip + knee + ankle = 0).
    The root height puts the undeformed soles at z = 0.
    """
    k = np.deg2rad(knee_flex_deg)

    def make(a):
        q = human.neutral()
        for s in ("L", "R"):
            q[human.dof(f"hip_{s}_ry")] = -a
            q[human.dof(f"knee_{s}_ry")] = k
            q[human.dof(f"ankle_{s}_ry")] = a - k
            q[human.dof(f"elbow_{s}_ry")] = -np.deg2rad(elbow_flex_deg)
        return q

    def err(a):
        q = make(a)
        pin.forwardKinematics(human.model, human.data, q)
        ank = human.data.oMi[human.joint_id["foot_L"]].translation[0]
        return human.com(q)[0] - ank - com_ahead_of_ankle

    a = brentq(err, -0.5, 0.8)
    q = make(a)
    pin.forwardKinematics(human.model, human.data, q)
    M = human.data.oMi[human.joint_id["foot_L"]]
    sole = (human.contact_local[0] @ M.rotation.T + M.translation)[:, 2].min()
    q[2] = -sole
    return q
