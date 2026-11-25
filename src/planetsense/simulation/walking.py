"""Planar (sagittal) walking with a finite-state balance controller.

The walker is the articulated model restricted to the sagittal plane (root
x, z, pitch and every flexion axis), with passive axial leg compliance.  The
gait is NOT prescribed: joint torques come from a state machine with
proportional-derivative targets and a balance feedback on swing-foot
placement (the structure of Yin, Loken & van de Panne's SIMBICON controller,
2007), and the resulting motion, contact timing and ground forces are whatever
the equations of motion produce.  The controller parameters are optimised
separately for each gravity (experiments/phase_C_walking).

Sign conventions inside the controller are "flexion positive"; they are mapped
to the model's right-hand RY coordinates here:
    thigh world flexion  phi   = -(pitch + hip_ry)
    knee flexion         kappa = +knee_ry
    foot world pitch     = pitch + hip_ry + knee_ry + ankle_ry   (toes down positive)

STATUS: a control hypothesis, not a model of human motor control.  A gait this
controller produces is credible only to the extent that its speed, cadence,
duty factor and ground-force shape agree with human data (checked for Earth in
docs/model_verification.md); for Moon and Mars it is a simulation-based
extrapolation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pinocchio as pin

from ..contact.compliant import ContactParams, static_penetration
from ..coupling.system import CoupledSystem
from ..human.control import LegSpring
from ..human.model import build_human

PARAM_NAMES = ("T_ds", "T0", "hip0", "hip1", "knee0", "knee1", "st_knee", "k_knee", "st_ankle",
               "k_ank", "foot_pitch", "lean", "c_d", "c_v", "push", "po_ankle", "gain", "arm")
# parameters that carry time, torque or velocity units and are rescaled by
# ``similarity_scaled``; all others are angles or dimensionless
_TIME_PARAMS = ("T_ds", "T0")


@dataclass
class WalkParams:
    T_ds: float = 0.10        # s, double support: trailing leg keeps pushing before it is lifted
    T0: float = 0.22          # s, duration of the swing-up state
    hip0: float = 0.35        # rad, swing thigh world flexion target, swing-up
    hip1: float = -0.20       # rad, base of the reach target (the feedback c_d * d is added)
    knee0: float = 1.00       # rad, swing knee flexion, swing-up
    knee1: float = 0.08       # rad, swing knee flexion, reach
    st_knee: float = 0.08     # rad, stance knee flexion target
    k_knee: float = 1.0       # multiplier on the stance knee stiffness
    st_ankle: float = 0.00    # rad, stance ankle spring rest angle (dorsiflexion positive)
    k_ank: float = 1.0        # multiplier on the stance ankle stiffness ("ankle rocker" spring)
    foot_pitch: float = -0.10 # rad, swing foot world pitch at landing (negative = toes up)
    lean: float = 0.05        # rad, forward lean of the pelvis
    c_d: float = 1.8          # rad/m, reach state: feedback on centre-of-mass position
    c_v: float = 0.20         # rad s/m, swing-up state: feedback on centre-of-mass velocity
    push: float = 0.0         # N m at Earth weight, extra plantar-flexion torque of the trailing ankle
    po_ankle: float = 0.0     # rad, plantar-flexion of the trailing ankle's rest angle in double support (push-off)
    gain: float = 1.0         # multiplier on all leg and trunk gains
    arm: float = 0.5          # arm swing as a fraction of the ipsilateral thigh angle

    def vector(self):
        return np.array([getattr(self, k) for k in PARAM_NAMES], float)

    @staticmethod
    def from_vector(x):
        return WalkParams(**{k: float(v) for k, v in zip(PARAM_NAMES, x)})

    asdict = asdict

    def similarity_scaled(self, g: float, g_ref: float = 9.81):
        """Dynamic-similarity transfer of a controller tuned at ``g_ref`` to
        gravity ``g`` (equal Froude number): times stretch by sqrt(g_ref/g),
        the velocity feedback gain (rad per m/s) likewise; angles are kept.
        Joint gains and torques are rescaled inside WalkController through
        ``g_gain``.  This is a labelled counterfactual, not a prediction."""
        r = float(np.sqrt(g_ref / g))
        d = asdict(self)
        for k in _TIME_PARAMS:
            d[k] *= r
        d["c_v"] *= r
        return WalkParams(**d)


# base gains at Earth gravity (N m/rad, N m s/rad) and torque limits (N m).
# Stance joints are stiff (they carry the body); the swing leg is driven softly,
# so that it moves almost ballistically and the hip couple does not load the
# stance leg (peak human hip and knee moments in swing are well below 1 N m/kg).
_GAINS = dict(hip=(0.0, 0.0, 150.0), hip_sw=(120.0, 25.0, 80.0),
              knee=(300.0, 25.0, 250.0), knee_sw=(60.0, 6.0, 60.0),
              ankle=(150.0, 10.0, 180.0), ankle_sw=(40.0, 3.0, 40.0),
              torso=(900.0, 90.0, 150.0), lumbar=(500.0, 40.0, 200.0), neck=(40.0, 3.0, 30.0),
              shoulder=(20.0, 2.0, 30.0), elbow=(10.0, 1.0, 20.0), wrist=(2.0, 0.1, 5.0))


class WalkController:
    """Three states per step, for the leg that is about to swing:

    0  double support  the trailing leg is still on the ground; its ankle
                       spring unloads (push-off) while the knee is released
    1  swing-up        timer; thigh flexes, knee folds, foot clears the ground
    2  reach           until the foot touches down; thigh target follows the
                       centre of mass (foot placement feedback), knee extends
    """
    wants_system = True
    n_state = 0

    def __init__(self, human, params: WalkParams, leg: LegSpring, g_gain: float = 9.81,
                 contact_fraction: float = 0.05):
        self.h, self.p, self.leg = human, params, leg
        d = human.dof
        self.ip = d("root_ry")
        self.ix = d("root_x")
        self.hip = {s: d(f"hip_{s}_ry") for s in "LR"}
        self.knee = {s: d(f"knee_{s}_ry") for s in "LR"}
        self.ank = {s: d(f"ankle_{s}_ry") for s in "LR"}
        self.legz = {s: d(f"ankle_{s}_z") for s in "LR"}
        self.sho = {s: d(f"shoulder_{s}_ry") for s in "LR"}
        self.elb = {s: d(f"elbow_{s}_ry") for s in "LR"}
        self.wri = {s: d(f"wrist_{s}_ry") for s in "LR"}
        self.ilum, self.inek = d("lumbar_ry"), d("neck_ry")
        self.foot = {"L": 0, "R": 1}
        # gains scale with the gravity they are designed for (weight-bearing torques ~ g)
        self.gr = g_gain / 9.81
        sc = params.gain * self.gr
        self.G = {k: (kp * sc, kd * np.sqrt(sc), tm * max(self.gr, 0.35)) for k, (kp, kd, tm) in _GAINS.items()}
        self.f_th = contact_fraction * human.total_mass * human.g
        # indices of the torque-driven coordinates (everything except the passive leg springs)
        self.i_act = np.array([i for i in range(human.n_root, human.nv)
                               if i not in self.legz.values()]) - human.n_root
        self.reset()

    def reset(self, stance="L", t=0.0):
        self.stance, self.phase, self.t_phase = stance, 0, t
        self.events = []          # (time, new stance leg, stance ankle x)

    # ------------------------------------------------------------ state machine
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
                self.events.append((t, sw, float(sys._p[i, 0])))

    def torque_sys(self, t, q, v, sys, first):
        if first:
            self._fsm(t, sys)
        h, p, G = self.h, self.p, self.G
        st = self.stance
        sw = "R" if st == "L" else "L"
        ph = self.phase
        ql, vl = q.tolist(), v.tolist()                 # plain floats: this runs four times per step
        tau = [0.0] * len(ql)

        def pd(key, target, i, extra=0.0, ks=1.0):
            kp, kd, tm = G[key]
            x = ks * kp * (target - ql[i]) - kd * vl[i] + extra
            return -tm if x < -tm else (tm if x > tm else x)

        th, om = ql[self.ip], vl[self.ip]
        pin.centerOfMass(h.model, h.data, q, v)
        cx, cv = float(h.data.com[0][0]), float(h.data.vcom[0][0])
        d = cx - float(sys._p[self.foot[st], 0])
        ih_sw, ih_st = self.hip[sw], self.hip[st]
        # ---- swing (or trailing) leg
        if ph == 2:
            phi_d = p.hip1 + p.c_d * d                  # reach: follow the centre of mass
        else:
            phi_d = p.hip0 + p.c_v * cv                 # lift on a timer
        phi = -(th + ql[ih_sw])
        dphi = -(om + vl[ih_sw])
        kp, kd, tm = G["hip_sw"]
        t_sw = -(kp * (phi_d - phi) - kd * dphi)         # RY torque on the swing thigh
        t_sw = -tm if t_sw < -tm else (tm if t_sw > tm else t_sw)
        kpt, kdt, tmt = G["torso"]
        t_to = kpt * (p.lean - th) - kdt * om            # pelvis pitch, applied through the stance hip
        t_to = -tmt if t_to < -tmt else (tmt if t_to > tmt else t_to)
        t_st = -t_to - t_sw
        tm = G["hip"][2]
        tau[ih_sw] = t_sw
        tau[ih_st] = -tm if t_st < -tm else (tm if t_st > tm else t_st)
        if ph == 0:      # trailing foot still loaded: ankle spring keeps pushing, knee is free
            tau[self.knee[sw]] = pd("knee_sw", 0.0, self.knee[sw], ks=0.0)
            tau[self.ank[sw]] = pd("ankle", -p.st_ankle + p.po_ankle, self.ank[sw], p.push * self.gr, p.k_ank)
        else:            # knee folds then extends; foot held at a world pitch
            tau[self.knee[sw]] = pd("knee_sw", p.knee0 if ph == 1 else p.knee1, self.knee[sw])
            a_d = p.foot_pitch - (th + ql[ih_sw] + ql[self.knee[sw]])
            tau[self.ank[sw]] = pd("ankle_sw", a_d, self.ank[sw])
        # ---- stance leg
        tau[self.knee[st]] = pd("knee", p.st_knee, self.knee[st], ks=p.k_knee)
        tau[self.ank[st]] = pd("ankle", -p.st_ankle, self.ank[st], 0.0, p.k_ank)
        # ---- trunk, head, arms
        tau[self.ilum] = pd("lumbar", 0.0, self.ilum)
        tau[self.inek] = pd("neck", 0.0, self.inek)
        for s_ in "LR":
            tau[self.sho[s_]] = pd("shoulder", -p.arm * (th + ql[self.hip[s_]]) - th, self.sho[s_])
            tau[self.elb[s_]] = pd("elbow", -0.35, self.elb[s_])
            tau[self.wri[s_]] = pd("wrist", 0.0, self.wri[s_])
        # passive axial leg springs (same law as LegSpring.force, scalar form)
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


def initial_state(human, leg: LegSpring, contact: ContactParams, x0: float, v0: float,
                  knee: float = 0.08):
    """Upright start with both feet on the ground, legs and pads pre-compressed
    to their static values, moving forward at ``v0``."""
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
    v = np.zeros(human.nv)
    v[human.dof("root_x")] = v0
    return q, v


def make_walker(g: float, params: WalkParams, leg: LegSpring, deck=None,
                contact: ContactParams | None = None, x0: float = 0.0, v0: float = 0.6,
                y_walk: float = 0.0, shaker_xy=None, g_gain: float | None = None):
    contact = contact or ContactParams()
    human = build_human(g=g, sagittal=True, leg_axial=True)
    ctrl = WalkController(human, params, leg, g_gain=g if g_gain is None else g_gain)
    sys = CoupledSystem(human, contact, ctrl, deck=deck, shaker_xy=shaker_xy)
    sys.y_shift = y_walk
    q, v = initial_state(human, leg, contact, x0, v0)
    y = sys.pack(q, v)
    sys.reset_anchors(y)
    return sys, y


def gait_metrics(out, human, t_skip: float = 3.0, i_act=None):
    """Spatiotemporal and force descriptors of a walking record (steady part)."""
    t = out["t"]
    W = human.total_mass * human.g
    fz = out["fc"][:, :, 2]                       # (n, 2) vertical force on each foot
    on = fz > 0.02 * W
    m = t >= t_skip
    res = dict(fell=bool(out["fell"]), duration=float(t[-1]))
    if m.sum() < 10 or out["fell"]:
        return res
    cx = out["com"][:, 0]
    res["speed"] = float((cx[m][-1] - cx[m][0]) / (t[m][-1] - t[m][0]))
    # touchdown events per foot.  A contact counts as a stance (and its start
    # as a touchdown) only if it lasts at least min_contact; shorter contacts
    # are swing-phase scuffs (a toe brushing the ground), counted separately.
    strikes = []
    min_contact = 0.08
    n_scuff = 0
    for i in (0, 1):
        k_up = np.nonzero(~on[:-1, i] & on[1:, i])[0] + 1
        k_dn = np.nonzero(on[:-1, i] & ~on[1:, i])[0] + 1
        for j in k_up:
            nxt = k_dn[k_dn > j]
            dur = (t[nxt[0]] - t[j]) if len(nxt) else np.inf
            if t[j] < t_skip:
                continue
            if dur >= min_contact:
                strikes.append((t[j], i))
            else:
                n_scuff += 1
    strikes.sort()
    if len(strikes) < 4:
        return res
    ts = np.array([s[0] for s in strikes])
    alt = [s[1] for s in strikes]
    res["n_steps"] = len(strikes)
    res["swing_scuffs_per_step"] = float(n_scuff / max(len(strikes), 1))
    res["alternating"] = bool(all(a != b for a, b in zip(alt[:-1], alt[1:])))
    dts = np.diff(ts)
    res["step_time"] = float(dts.mean())
    res["step_time_cv"] = float(dts.std() / dts.mean())
    res["step_frequency"] = float(1.0 / dts.mean())
    res["step_length"] = float(res["speed"] * dts.mean())
    mm = (t >= ts[0]) & (t <= ts[-1])
    res["duty_factor"] = float(on[mm].mean())
    both = on[mm].all(axis=1)
    none = (~on[mm]).all(axis=1)
    res["double_support_fraction"] = float(both.mean())
    res["flight_fraction"] = float(none.mean())
    tot = fz[mm].sum(axis=1) / W
    res["grf_peak_over_W"] = float(tot.max())
    res["grf_min_over_W"] = float(tot.min())
    res["grf_mean_over_W"] = float(tot.mean())
    # dynamic load factors of the total vertical force at the first three step harmonics
    tt, x = t[mm], tot - tot.mean()
    fp = res["step_frequency"]
    res["dlf"] = [float(2 * abs(np.mean(x * np.exp(-2j * np.pi * k * fp * tt)))) for k in (1, 2, 3)]
    res["grf_foot_peak_over_W"] = float(fz[mm].max() / W)
    res["com_vertical_excursion"] = float(np.ptp(out["com"][mm, 2]))
    tq = out["tau"][mm] if i_act is None else out["tau"][mm][:, i_act]
    res["peak_torque"] = float(np.abs(tq).max())
    # dimensionless effort: mean square joint torque relative to (m g L)^2, L = 1 m
    res["effort"] = float(np.mean(np.sum(tq ** 2, axis=1)) / W ** 2)
    # the same with Earth weight as the unit, so that planets can be compared
    res["effort_abs"] = float(np.mean(np.sum(tq ** 2, axis=1)) / (human.total_mass * 9.81) ** 2)
    dist = cx[mm][-1] - cx[mm][0]
    e = out["energy"]
    work_pos = np.sum(np.maximum(np.diff(e[mm, 4]), 0.0))        # positive joint work (incl. passive springs)
    res["positive_work_per_kg_m"] = float(work_pos / (human.total_mass * max(dist, 1e-6)))
    # energy audit: drift of the work-energy identity relative to the work exchanged
    bal = e[:, 8]
    through = np.abs(np.diff(e[:, 4])).sum() + np.abs(np.diff(e[:, 5])).sum() + 1e-9
    res["energy_drift_rel"] = float(np.abs(bal - bal[0]).max() / through)
    res["gait"] = ("walk" if res["flight_fraction"] < 0.02 and res["duty_factor"] > 0.5
                   else "bouncing (flight phases)")
    # forward lean of the trunk (pelvis pitch + lumbar flexion, rad, positive forward; needs the
    # recorded coordinates): mean over the steady part, and its range within the gait cycle
    if "q" in out:
        th = out["q"][:, human.dof("root_ry")] + out["q"][:, human.dof("lumbar_ry")]
        res["trunk_lean"] = float(th[mm].mean())
        res["trunk_lean_range"] = float(np.ptp(th[mm]))
    return res
