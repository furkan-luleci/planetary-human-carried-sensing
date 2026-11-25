"""Three-dimensional walking: the full 38-coordinate body with a balance
controller in both the sagittal and the frontal plane.

Same principle as the planar walker (planetsense.simulation.walking): the gait is
not prescribed; a finite-state controller sets joint targets, and steps,
forces and body motion come out of the equations of motion.  Added in 3-D
(the structure of the original 3-D SIMBICON controller, Yin, Loken & van de
Panne 2007):

* swing-leg placement in BOTH planes: the swing thigh's target direction in
  the walker's heading frame has a forward angle (fed back from the centre of
  mass's fore-aft position and velocity, as in 2-D) and a sideways angle
  (fed back from its lateral position and velocity relative to the stance
  ankle); the target direction is converted to hip joint angles through the
  current pelvis orientation;
* pelvis orientation (pitch = lean, roll = 0, yaw = heading 0) held by the
  STANCE hip in world coordinates: the stance hip supplies the torque the
  pelvis needs minus what the swing hip already applies to it;
* sideways ankle stiffness of the stance foot, swing foot held level,
  arm swing with slight abduction, trunk and head held upright.

STATUS: a control hypothesis tuned to gait descriptors (Phase G), not a model
of human motor control.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pinocchio as pin

from ..contact.compliant import ContactParams, static_penetration
from ..coupling.system import CoupledSystem
from ..human.control import LegSpring
from ..human.model import build_human

PARAM3D = ("T_ds", "T0", "hip0", "hip1", "knee0", "knee1", "st_knee", "k_knee", "st_ankle", "k_ank",
           "foot_pitch", "lean", "c_d", "c_v", "push", "po_ankle", "gain", "arm",
           "w0", "cd_lat", "cv_lat", "k_ank_lat", "k_hip_lat", "arm_abd", "zab", "lam_lat", "k_yaw", "brake", "tau_head",
           "lumbar_ext")
_TIME = ("T_ds", "T0")


@dataclass
class WalkParams3D:
    # sagittal (same meaning as WalkParams)
    T_ds: float = 0.10
    T0: float = 0.22
    hip0: float = 0.35
    hip1: float = -0.20
    knee0: float = 1.00
    knee1: float = 0.08
    st_knee: float = 0.08
    k_knee: float = 1.0
    st_ankle: float = 0.00
    k_ank: float = 1.0
    foot_pitch: float = -0.10
    lean: float = 0.05
    c_d: float = 1.8
    c_v: float = 0.20
    push: float = 0.0
    po_ankle: float = 0.0
    gain: float = 1.0
    arm: float = 0.5
    # frontal plane
    w0: float = 0.0           # rad, base sideways angle of the swing thigh (outward positive): sets step width
    cd_lat: float = 0.8       # rad/m, sideways placement feedback on the centre of mass's lateral offset
    cv_lat: float = 0.3       # rad s/m, ... on its lateral velocity
    k_ank_lat: float = 1.0    # multiplier on the stance ankle's sideways stiffness
    k_hip_lat: float = 1.0    # multiplier on the swing hip's sideways gains
    arm_abd: float = 0.10     # rad, arm abduction
    # terminal swing and pelvis rotation (defaults reproduce the first Stage-1 controller)
    zab: float = 1.0          # multiplier on the swing hip's sideways damping only
    lam_lat: float = 1.0      # share of the sideways placement feedback still followed during the reach phase
                              # (0: placement held at the value chosen when the reach phase starts)
    k_yaw: float = 1.0        # multiplier on the stance hip's axial-rotation gains (pelvis yaw relative to the planted foot)
    brake: float = 0.0        # N s/m, terminal swing: damping of the swing foot's sideways velocity over the
                              # ground during the reach phase (applied through the swing hip's sideways axis)
    lumbar_ext: float = 0.0   # rad, extension target of the lower back (trunk tilted back relative to the pelvis);
                              # lets the trunk stand upright while the pelvis keeps a forward tilt
    tau_head: float = 0.0     # s, time constant of the heading used for foot placement: 0 = the pelvis's
                              # instantaneous yaw (first Stage-1 controller); > 0 = the pelvis yaw low-pass
                              # filtered, i.e. the walking direction without the stride-to-stride pelvis rotation

    def vector(self):
        return np.array([getattr(self, k) for k in PARAM3D], float)

    @staticmethod
    def from_vector(x):
        return WalkParams3D(**{k: float(v) for k, v in zip(PARAM3D, x)})

    asdict = asdict

    def similarity_scaled(self, g: float, g_ref: float = 9.81):
        r = float(np.sqrt(g_ref / g))
        d = asdict(self)
        for k in _TIME:
            d[k] *= r
        d["c_v"] *= r
        d["cv_lat"] *= r
        return WalkParams3D(**d)


# base gains at Earth gravity (N m/rad, N m s/rad, torque limit N m)
_G3 = dict(hip_sw=(120.0, 25.0, 80.0), hip_sw_ab=(200.0, 20.0, 100.0), hip_rot=(60.0, 6.0, 40.0),
           hip_st=(0.0, 0.0, 250.0),
           knee=(300.0, 25.0, 250.0), knee_sw=(60.0, 6.0, 60.0),
           ankle=(150.0, 10.0, 180.0), ankle_sw=(40.0, 3.0, 40.0),
           ankle_rx=(150.0, 8.0, 120.0), ankle_rx_sw=(30.0, 2.0, 30.0),
           lumbar=(500.0, 40.0, 200.0), neck=(40.0, 3.0, 30.0),
           shoulder=(20.0, 2.0, 30.0), elbow=(10.0, 1.0, 20.0), wrist=(2.0, 0.1, 5.0))
# pelvis attitude gains (world roll x, pitch y; yaw is not controlled through the stance hip)
_PELVIS_KP = np.array([900.0, 900.0, 300.0])
_PELVIS_KD = np.array([90.0, 90.0, 30.0])
_EY, _EX, _EZ = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class WalkController3D:
    wants_system = True
    n_state = 0

    # start-up aid: springs on the pelvis's lateral position, roll and heading
    # during gait initiation only (full until t1, fading to zero at t2).  It is a
    # generalised force on the root coordinates, so its work is part of the
    # energy audit.  Walkers start >= 4 s before the bridge, so it never acts there.
    HELP_T1, HELP_T2 = 2.0, 3.0
    HELP_K = (2000.0, 1500.0, 600.0)     # N/m, N m/rad, N m/rad  (lateral, roll, yaw)
    HELP_C = (400.0, 150.0, 60.0)

    def __init__(self, human, params: WalkParams3D, leg: LegSpring, g_gain: float = 9.81,
                 contact_fraction: float = 0.05, y_ref: float = 0.0):
        self.h, self.p, self.leg = human, params, leg
        self.y_ref = y_ref
        self.t0 = None
        d = human.dof
        self.iroot = {k: d(f"root_{k}") for k in ("x", "y", "z", "rz", "ry", "rx")}
        S = "LR"
        self.hip = {s: (d(f"hip_{s}_ry"), d(f"hip_{s}_rx"), d(f"hip_{s}_rz")) for s in S}
        self.knee = {s: d(f"knee_{s}_ry") for s in S}
        self.ank = {s: (d(f"ankle_{s}_ry"), d(f"ankle_{s}_rx")) for s in S}
        self.legz = {s: d(f"ankle_{s}_z") for s in S}
        self.sho = {s: (d(f"shoulder_{s}_ry"), d(f"shoulder_{s}_rx"), d(f"shoulder_{s}_rz")) for s in S}
        self.elb = {s: d(f"elbow_{s}_ry") for s in S}
        self.wri = {s: (d(f"wrist_{s}_rx"), d(f"wrist_{s}_ry")) for s in S}
        self.lum = (d("lumbar_ry"), d("lumbar_rx"), d("lumbar_rz"))
        self.nek = (d("neck_ry"), d("neck_rx"), d("neck_rz"))
        self.side = {"L": 1.0, "R": -1.0}
        self.foot = {"L": 0, "R": 1}
        self.jp = human.joint_id["pelvis"]
        self.jfoot = {s: human.joint_id[f"foot_{s}"] for s in S}
        self.jthigh = {s: human.joint_id[f"thigh_{s}"] for s in S}
        self.gr = g_gain / 9.81
        sc = params.gain * self.gr
        G = {k: (kp * sc, kd * math.sqrt(sc), tm * max(self.gr, 0.35)) for k, (kp, kd, tm) in _G3.items()}
        for k in ("hip_sw_ab",):
            kp, kd, tm = G[k]
            G[k] = (kp * params.k_hip_lat, kd * math.sqrt(params.k_hip_lat) * params.zab, tm)
        kp, kd, tm = G["hip_rot"]
        G["hip_rot_st"] = (kp * params.k_yaw, kd * math.sqrt(params.k_yaw), tm)
        self.G = G
        self.pkp = _PELVIS_KP * sc
        self.pkd = _PELVIS_KD * math.sqrt(sc)
        self.f_th = contact_fraction * human.total_mass * human.g
        self.i_act = np.array([i for i in range(human.n_root, human.nv)
                               if i not in self.legz.values()]) - human.n_root
        self.reset()

    def reset(self, stance="L", t=0.0):
        self.stance, self.phase, self.t_phase = stance, 0, t
        self.events = []          # (time, new stance foot, ankle x, ankle y)
        self._frz_t, self._frz = None, 0.0
        self._head, self._t_head = None, None

    def _fsm(self, t, sys):
        sw = "R" if self.stance == "L" else "L"
        if self.phase == 0:
            if t - self.t_phase >= self.p.T_ds:
                self.phase, self.t_phase = 1, t
        elif self.phase == 1:
            if t - self.t_phase >= self.p.T0:
                self.phase, self.t_phase = 2, t
        else:
            i = self.foot[sw]
            if t - self.t_phase > 0.02 and sys._fn[sys.npf * i:sys.npf * (i + 1)].sum() > self.f_th:
                self.stance, self.phase, self.t_phase = sw, 0, t
                self.events.append((t, sw, float(sys._p[i, 0]), float(sys._p[i, 1])))

    @staticmethod
    def _hip_axes(Rp, a, b):
        """World directions of the hip's elementary axes (RY, RX, RZ in the pelvis frame)."""
        R1 = Rp @ _ry(a)
        R2 = R1 @ _rx(b)
        return np.column_stack([Rp @ _EY, R1 @ _EX, R2 @ _EZ])

    def torque_sys(self, t, q, v, sys, first):
        if first:
            self._fsm(t, sys)
        h, p, G = self.h, self.p, self.G
        st = self.stance
        sw = "R" if st == "L" else "L"
        ph = self.phase
        ql, vl = q.tolist(), v.tolist()
        tau = [0.0] * len(ql)

        def pd(key, target, i, extra=0.0, ks=1.0):
            kp, kd, tm = G[key]
            x = ks * kp * (target - ql[i]) - kd * vl[i] + extra
            return -tm if x < -tm else (tm if x > tm else x)

        d = h.data
        Mp = d.oMi[self.jp]
        Rp = Mp.rotation
        w_loc = d.v[self.jp].angular                      # pelvis angular velocity, pelvis frame
        w_world = Rp @ w_loc
        yaw = ql[self.iroot["rz"]]
        pitch = ql[self.iroot["ry"]]
        roll = ql[self.iroot["rx"]]
        pin.centerOfMass(h.model, d, q, v)
        com, vcom = d.com[0], d.vcom[0]
        if p.tau_head > 0.0:
            # heading for foot placement: the pelvis yaw low-pass filtered (updated once per time step)
            if self._head is None:
                self._head, self._t_head = yaw, t
            elif first:
                self._head += min(1.0, (t - self._t_head) / p.tau_head) * (yaw - self._head)
                self._t_head = t
            Rh = _rz(self._head)
        else:
            Rh = _rz(yaw)                                  # heading frame
        ia = self.foot[st]
        rel = Rh.T @ (com - sys._p[ia])                    # centre of mass relative to the stance ankle, heading frame
        vh = Rh.T @ vcom
        # ---------------------------------------------------------------- swing leg
        sd = self.side[sw]
        if ph == 2:
            phi_s = p.hip1 + p.c_d * rel[0]
        else:
            phi_s = p.hip0 + p.c_v * vh[0]
        phi_c = sd * p.w0 + p.cd_lat * rel[1] + p.cv_lat * vh[1]       # signed sideways angle, +y = left
        if ph == 2 and p.lam_lat != 1.0:
            # terminal swing: the sideways placement chosen when the reach phase starts is
            # (partly) held, so the foot is not still chasing the centre of mass at touchdown
            if self._frz_t != self.t_phase:
                self._frz_t, self._frz = self.t_phase, phi_c
            phi_c = self._frz + p.lam_lat * (phi_c - self._frz)
        cc = math.cos(phi_c)
        dirw = Rh @ np.array([math.sin(phi_s) * cc, math.sin(phi_c), -math.cos(phi_s) * cc])
        u = Rp.T @ dirw
        a_d = math.atan2(-u[0], -u[2])
        b_d = math.asin(max(-1.0, min(1.0, u[1])))
        ih = self.hip[sw]
        a, b, c = ql[ih[0]], ql[ih[1]], ql[ih[2]]
        kp, kd, tm = G["hip_sw"]
        t_a = kp * (a_d - a) - kd * (vl[ih[0]] + w_loc[1])
        t_a = -tm if t_a < -tm else (tm if t_a > tm else t_a)
        kp, kd, tm = G["hip_sw_ab"]
        t_b = kp * (b_d - b) - kd * (vl[ih[1]] + w_loc[0])
        if ph == 2 and p.brake > 0.0:
            # terminal swing: the foot's sideways velocity over the ground is damped (ramped in over 50 ms)
            jf = self.jfoot[sw]
            Mf = d.oMi[jf]
            v_lat = float((Rh.T @ (Mf.rotation @ d.v[jf].linear))[1])
            L_leg = float(np.linalg.norm(d.oMi[self.jthigh[sw]].translation - Mf.translation))
            t_b -= min(1.0, (t - self.t_phase) / 0.05) * p.brake * self.gr * L_leg * v_lat
        t_b = -tm if t_b < -tm else (tm if t_b > tm else t_b)
        kp, kd, tm = G["hip_rot"]
        # swing foot lands pointing along the walking direction (world yaw 0): heading is
        # corrected step by step through foot placement, not by twisting the planted leg
        c_d = max(-0.3, min(0.3, -yaw))
        t_c = kp * (c_d - c) - kd * vl[ih[2]]
        t_c = -tm if t_c < -tm else (tm if t_c > tm else t_c)
        tau[ih[0]], tau[ih[1]], tau[ih[2]] = t_a, t_b, t_c
        S_sw = self._hip_axes(Rp, a, b)
        T_sw_on_pelvis = -(S_sw @ np.array([t_a, t_b, t_c]))
        # ---------------------------------------------------------------- pelvis attitude via the stance hip
        Rd = _ry(p.lean)                                   # heading 0, roll 0, forward lean
        e = pin.log3(Rd @ Rp.T)
        T_pelvis = self.pkp * e - self.pkd * w_world
        T_pelvis[2] = 0.0                                  # no world-yaw torque through the planted leg
        ist = self.hip[st]
        S_st = self._hip_axes(Rp, ql[ist[0]], ql[ist[1]])
        try:
            t_st = -np.linalg.solve(S_st, T_pelvis - T_sw_on_pelvis)
        except np.linalg.LinAlgError:
            t_st = np.zeros(3)
        # axial rotation of the stance leg held relative to the pelvis (joint space)
        kp, kd, _ = G["hip_rot_st"]
        t_st[2] += kp * (0.0 - ql[ist[2]]) - kd * vl[ist[2]]
        tm = G["hip_st"][2]
        for k in range(3):
            x = float(t_st[k])
            tau[ist[k]] = -tm if x < -tm else (tm if x > tm else x)
        # ---------------------------------------------------------------- knees and ankles
        ia_sw, ia_st = self.ank[sw], self.ank[st]
        if ph == 0:      # trailing foot still loaded: push-off, knee released
            tau[self.knee[sw]] = pd("knee_sw", 0.0, self.knee[sw], ks=0.0)
            tau[ia_sw[0]] = pd("ankle", -p.st_ankle + p.po_ankle, ia_sw[0], p.push * self.gr, p.k_ank)
            tau[ia_sw[1]] = pd("ankle_rx", 0.0, ia_sw[1], 0.0, p.k_ank_lat)
        else:
            tau[self.knee[sw]] = pd("knee_sw", p.knee0 if ph == 1 else p.knee1, self.knee[sw])
            # swing foot: world pitch held at foot_pitch, sole kept level sideways
            pitch_leg = pitch + a + ql[self.knee[sw]]
            tau[ia_sw[0]] = pd("ankle_sw", p.foot_pitch - pitch_leg, ia_sw[0])
            tau[ia_sw[1]] = pd("ankle_rx_sw", -(roll + b), ia_sw[1])
        tau[self.knee[st]] = pd("knee", p.st_knee, self.knee[st], ks=p.k_knee)
        tau[ia_st[0]] = pd("ankle", -p.st_ankle, ia_st[0], 0.0, p.k_ank)
        tau[ia_st[1]] = pd("ankle_rx", 0.0, ia_st[1], 0.0, p.k_ank_lat)
        # ---------------------------------------------------------------- trunk, head, arms
        for i in self.lum:
            tau[i] = pd("lumbar", -p.lumbar_ext if i == self.lum[0] else 0.0, i)
        for i in self.nek:
            tau[i] = pd("neck", 0.0, i)
        for s_ in "LR":
            sh = self.sho[s_]
            tau[sh[0]] = pd("shoulder", -p.arm * (pitch + ql[self.hip[s_][0]]) - (pitch - p.lumbar_ext), sh[0])
            tau[sh[1]] = pd("shoulder", self.side[s_] * p.arm_abd, sh[1])
            tau[sh[2]] = pd("shoulder", 0.0, sh[2])
            tau[self.elb[s_]] = pd("elbow", -0.35, self.elb[s_])
            for i in self.wri[s_]:
                tau[i] = pd("wrist", 0.0, i)
        # start-up aid (see class attributes)
        if self.t0 is None:
            self.t0 = t
        ts = t - self.t0
        if ts < self.HELP_T2:
            f = 1.0 if ts < self.HELP_T1 else (self.HELP_T2 - ts) / (self.HELP_T2 - self.HELP_T1)
            ky, kr, kz = self.HELP_K
            cy, cr, cz = self.HELP_C
            iy, irx, irz = self.iroot["y"], self.iroot["rx"], self.iroot["rz"]
            tau[iy] += f * (-ky * (ql[iy] - self.y_ref) - cy * vl[iy])
            tau[irx] += f * (-kr * ql[irx] - cr * vl[irx])
            tau[irz] += f * (-kz * ql[irz] - cz * vl[irz])
        # passive axial leg springs
        lg = self.leg
        for s_ in "LR":
            i = self.legz[s_]
            dd, dv = ql[i], vl[i]
            if dd >= 0.0:
                el = lg.k * dd ** lg.n
                f = el + lg.c * dv if lg.damping == "constant" else el * (1.0 + lg.c * dv)
            else:
                f = lg.k_ext * dd + lg.c_ext * dv
            tau[i] -= f
        return np.array(tau)

    def elastic_energy(self, q):
        return self.leg.energy(q[[self.legz["L"], self.legz["R"]]])


def initial_state3d(human, leg: LegSpring, contact: ContactParams, x0: float, y0: float, v0: float,
                    knee: float = 0.08):
    q = human.neutral()
    W = human.total_mass * human.g
    dl = leg.static_compression(0.5 * W)
    dp = static_penetration(contact, W / 8)
    for s in "LR":
        q[human.dof(f"hip_{s}_ry")] = -0.5 * knee
        q[human.dof(f"knee_{s}_ry")] = knee
        q[human.dof(f"ankle_{s}_ry")] = -0.5 * knee
        q[human.dof(f"ankle_{s}_z")] = dl
        q[human.dof(f"elbow_{s}_ry")] = -0.35
    pin.forwardKinematics(human.model, human.data, q)
    M = human.data.oMi[human.joint_id["foot_L"]]
    sole = (human.contact_local[0] @ M.rotation.T + M.translation)[:, 2].min()
    q[human.dof("root_z")] = -sole - dp
    q[human.dof("root_x")] = x0
    q[human.dof("root_y")] = y0
    v = np.zeros(human.nv)
    v[human.dof("root_x")] = v0
    return q, v


def planar_seed(g: float, planar_params, leg: LegSpring, v0: float, T: float = 8.0, path: str | None = None,
                g_gain: float | None = None):
    """A state on the steady gait cycle of the PLANAR walker, taken at a
    touchdown (start of double support).  Used to start the 3-D walker without
    the violent first steps of a standing start.  Cached in ``path`` if given."""
    import os
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return dict(names=[str(n) for n in z["names"]], q=z["q"], v=z["v"], stance=str(z["stance"]))
    from .walking import make_walker
    sys_, y = make_walker(g, planar_params, leg, v0=v0, g_gain=g_gain)
    c = sys_.controller
    dt, t, nev = 2.5e-4, 0.0, 0
    best = None
    for i in range(int(T / dt)):
        y, _ = sys_.step(t, y, dt)
        t += dt
        if len(c.events) > nev:
            nev = len(c.events)
            if t > T - 2.0 and best is None:
                best = (y.copy(), c.stance)
        if y[sys_.iz] < 0.6 * 0.95:
            raise RuntimeError("planar walker fell while generating the seed")
    if best is None:
        raise RuntimeError("no touchdown in the seed window")
    yq = best[0]
    seed = dict(names=list(sys_.h.dof_names), q=yq[:sys_.nq].copy(), v=yq[sys_.nq:2 * sys_.nq].copy(), stance=best[1])
    if path:
        np.savez(path, names=np.array(seed["names"]), q=seed["q"], v=seed["v"], stance=np.array(seed["stance"]))
    return seed


def make_walker3d(g: float, params: WalkParams3D, leg: LegSpring, deck=None,
                  contact: ContactParams | None = None, x0: float = 0.0, v0: float = 0.6,
                  y_walk: float = 0.0, shaker_xy=None, g_gain: float | None = None, seed: dict | None = None):
    """``seed``: a planar gait-cycle state (``planar_seed``); its sagittal
    coordinates are copied, the walker is moved to (x0, y_walk), lateral
    coordinates start at zero, and the controller starts in double support with
    the seed's stance foot."""
    contact = contact or ContactParams()
    human = build_human(g=g, leg_axial=True)
    ctrl = WalkController3D(human, params, leg, g_gain=g if g_gain is None else g_gain, y_ref=y_walk)
    sys = CoupledSystem(human, contact, ctrl, deck=deck, shaker_xy=shaker_xy)
    if seed is None:
        q, v = initial_state3d(human, leg, contact, x0, y_walk, v0)
    else:
        q, v = human.neutral(), np.zeros(human.nv)
        for n, qq, vv in zip(seed["names"], seed["q"], seed["v"]):
            if n in human.dof_names:
                q[human.dof(n)], v[human.dof(n)] = qq, vv
        q[human.dof("root_x")] = x0
        q[human.dof("root_y")] = y_walk
        ctrl.reset(stance=seed["stance"], t=0.0)
    y = sys.pack(q, v)
    sys.reset_anchors(y)
    return sys, y


def lateral_metrics(out, human, events, t_skip: float = 3.0, step_frequency: float | None = None):
    """Frontal-plane gait descriptors: step width, lateral sway, lateral force, drift, heading."""
    t = out["t"]
    m = t >= t_skip
    res = {}
    ev = [e for e in events if e[0] >= t_skip]
    if len(ev) >= 3:
        ys = np.array([e[3] for e in ev])
        res["step_width"] = float(np.mean(np.abs(np.diff(ys))))
    if m.sum() > 10:
        W = human.total_mass * human.g
        cy = out["com"][m, 1]
        res["lateral_drift"] = float(abs(cy[-1] - cy[0]))
        res["com_lateral_excursion"] = float(np.ptp(cy - np.polyval(np.polyfit(t[m], cy, 1), t[m])))
        fy = out["fc"][m, :, 1].sum(axis=1) / W
        res["grf_lateral_peak_over_W"] = float(np.abs(fy).max())
        if step_frequency:
            fh = 0.5 * step_frequency                      # lateral force repeats once per stride
            res["dlf_lat"] = float(2 * abs(np.mean((fy - fy.mean()) * np.exp(-2j * np.pi * fh * t[m]))))
        res["heading_max"] = float(np.abs(out["root"][m, 3]).max())
    return res
