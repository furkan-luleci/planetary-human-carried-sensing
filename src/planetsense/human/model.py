"""Articulated rigid-body human (15 segments, 36 generalized coordinates).

Equations of motion

    M_h(q) vdot + h_h(q, v, g) = S^T tau + J_c(q)^T lambda_c

are evaluated with Pinocchio (articulated-body algorithm).  Every joint is a
sequence of elementary prismatic / revolute joints, so the configuration space
is a vector space and  qdot = v  holds exactly (no quaternion kinematics).
The price is an Euler-angle singularity when the middle axis of a 3-DOF joint
reaches +-90 degrees, far outside standing and walking postures.

Joint coordinates are right-hand rotations about segment axes (x anterior,
y left, z up in the neutral standing posture):

    root      x, y, z (world, m), yaw (z), pitch (y), roll (x)
    lumbar    RY, RX, RZ          neck      RY, RX, RZ
    shoulder  RY, RX, RZ          elbow     RY          wrist  RX, RY
    hip       RY, RX, RZ          knee      RY          ankle  RY, RX

With this convention hip flexion, knee extension beyond straight and ankle
dorsiflexion are NEGATIVE RY rotations of the distal segment; knee flexion is
positive RY.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from .anthropometry import build_segments, foot_contact_points, sensor_mounts

_ELEM = {"RX": pin.JointModelRX, "RY": pin.JointModelRY, "RZ": pin.JointModelRZ,
         "T": pin.JointModelTranslation, "PZ": pin.JointModelPZ, "PX": pin.JointModelPX}
_ORDER = ["pelvis", "torso", "head",
          "upper_arm_L", "forearm_L", "hand_L", "upper_arm_R", "forearm_R", "hand_R",
          "thigh_L", "shank_L", "foot_L", "thigh_R", "shank_R", "foot_R"]


def _joint_model(axes):
    if len(axes) == 1:
        return _ELEM[axes[0]]()
    jc = pin.JointModelComposite(_ELEM[axes[0]]())
    for a in axes[1:]:
        jc.addJoint(_ELEM[a]())
    return jc


def _dof_labels(joint, axes):
    out = []
    for a in axes:
        if a == "T":
            out += [f"{joint}_x", f"{joint}_y", f"{joint}_z"]
        elif a in ("PZ", "PX"):
            out.append(f"{joint}_{a[1].lower()}")
        else:
            out.append(f"{joint}_{a.lower()}")
    return out


@dataclass
class HumanModel:
    model: pin.Model
    data: pin.Data
    segments: dict
    joint_id: dict                 # segment name -> pinocchio joint id
    dof_names: list
    vslice: dict                   # joint name -> slice in v (== slice in q)
    contact_joint: np.ndarray      # (n_feet,) joint ids of the feet
    contact_local: np.ndarray      # (n_feet, n_pts, 3) contact points in foot frames
    foot_names: list
    sensor_frame: dict             # sensor name -> pinocchio frame id
    sensor_parent: dict            # sensor name -> segment name
    g: float
    meta: dict = field(default_factory=dict)

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def total_mass(self) -> float:
        return float(pin.computeTotalMass(self.model))

    @property
    def n_root(self) -> int:
        """Number of unactuated root coordinates (6 in 3D, 3 in the planar model)."""
        return int(self.meta.get("n_root", 6))

    @property
    def n_act(self) -> int:
        return self.model.nv - self.n_root

    def neutral(self) -> np.ndarray:
        return np.zeros(self.model.nq)

    def dof(self, name: str) -> int:
        return self.dof_names.index(name)

    def com(self, q, v=None):
        if v is None:
            return pin.centerOfMass(self.model, self.data, q).copy()
        pin.centerOfMass(self.model, self.data, q, v)
        return self.data.com[0].copy(), self.data.vcom[0].copy()

    def kinetic_energy(self, q, v) -> float:
        return float(pin.computeKineticEnergy(self.model, self.data, q, v))

    def potential_energy(self, q) -> float:
        """Gravitational potential energy m g z_com (zero at world z = 0)."""
        return float(pin.computePotentialEnergy(self.model, self.data, q))

    def set_gravity(self, g: float):
        self.g = float(g)
        self.model.gravity.linear = np.array([0.0, 0.0, -self.g])


def build_human(g: float = 9.81, body_mass: float = 75.0, stature: float = 1.75,
                geometry: dict | None = None, mass_scale: dict | None = None,
                added_inertia: dict | None = None, leg_axial: bool = False,
                sagittal: bool = False) -> HumanModel:
    """Assemble the articulated model.

    ``g`` is the magnitude of the gravitational acceleration used in the
    equations of motion; it does not change any mass or inertia.
    ``added_inertia`` maps a segment name to a list of
    ``(mass, com_in_segment_frame, inertia_3x3)`` rigidly attached bodies
    (hook for suit level L1: rigidly added segment masses, no double counting
    because the attached bodies are separate ``pin.Inertia`` terms).
    """
    seg = build_segments(body_mass, stature, geometry=geometry, mass_scale=mass_scale)
    if sagittal:
        # planar (sagittal) restriction: keep fore-aft and vertical translation
        # and every RY rotation; the 3D geometry and inertia are unchanged
        for name, sg in seg.items():
            sg.axes = ("PX", "PZ", "RY") if name == "pelvis" else tuple(a for a in sg.axes if a == "RY")
            if not sg.axes:
                sg.axes = ("FIXED",)
    if leg_axial:
        # axial ("telescoping") compliance of each leg: the foot can translate
        # along the shank axis; positive coordinate = leg shortening
        for side in ("L", "R"):
            seg[f"foot_{side}"].axes = ("PZ",) + tuple(seg[f"foot_{side}"].axes)
    m = pin.Model()
    m.name = "articulated_human_v1"
    m.gravity.linear = np.array([0.0, 0.0, -float(g)])
    jid, names, vsl = {}, [], {}
    place = {}
    for name in _ORDER:
        s = seg[name]
        if s.axes == ("FIXED",):
            # rigidly attached to its parent (hands in the planar model)
            pj, pT = place[s.parent]
            T = pT * pin.SE3(np.eye(3), s.offset)
            m.appendBodyToJoint(pj, pin.Inertia(s.mass, s.com, s.inertia), T)
            for (am, ac, aI) in (added_inertia or {}).get(name, []):
                m.appendBodyToJoint(pj, pin.Inertia(float(am), np.asarray(ac, float),
                                                    np.asarray(aI, float)), T)
            m.addFrame(pin.Frame(name, pj, T, pin.FrameType.BODY))
            jid[name] = pj
            place[name] = (pj, T)
            continue
        parent, pT = (0, pin.SE3.Identity()) if s.parent is None else place[s.parent]
        j = m.addJoint(parent, _joint_model(s.axes), pT * pin.SE3(np.eye(3), s.offset), s.joint)
        place[name] = (j, pin.SE3.Identity())
        m.appendBodyToJoint(j, pin.Inertia(s.mass, s.com, s.inertia), pin.SE3.Identity())
        for (am, ac, aI) in (added_inertia or {}).get(name, []):
            m.appendBodyToJoint(j, pin.Inertia(float(am), np.asarray(ac, float),
                                               np.asarray(aI, float)), pin.SE3.Identity())
        m.addFrame(pin.Frame(name, j, pin.SE3.Identity(), pin.FrameType.BODY))
        jid[name] = j
        names += _dof_labels(s.joint, s.axes)
        vsl[s.joint] = slice(m.joints[j].idx_v, m.joints[j].idx_v + m.joints[j].nv)
    assert m.nq == m.nv == len(names)
    cp = foot_contact_points(seg, geometry)
    feet = ["foot_L", "foot_R"]
    sens_f, sens_p = {}, {}
    for sname, (parent, pos) in sensor_mounts(seg).items():
        sens_f[sname] = m.addFrame(pin.Frame(sname, jid[parent], pin.SE3(np.eye(3), pos),
                                             pin.FrameType.OP_FRAME))
        sens_p[sname] = parent
    return HumanModel(model=m, data=m.createData(), segments=seg, joint_id=jid,
                      dof_names=names, vslice=vsl,
                      contact_joint=np.array([jid[f] for f in feet]),
                      contact_local=np.array([cp[f] for f in feet]),
                      foot_names=feet, sensor_frame=sens_f, sensor_parent=sens_p,
                      g=float(g),
                      meta=dict(body_mass=body_mass, stature=stature,
                                added_inertia=bool(added_inertia), leg_axial=leg_axial,
                                sagittal=sagittal, n_root=3 if sagittal else 6))


def to_mjcf(h: HumanModel, sensors: bool = True) -> str:
    """Equivalent MuJoCo model (no contact), used ONLY for cross-solver checks."""
    seg = h.segments
    kids: dict = {}
    for n in _ORDER:
        kids.setdefault(seg[n].parent, []).append(n)
    sens = {}
    for sname, fid in h.sensor_frame.items():
        sens.setdefault(h.sensor_parent[sname], []).append(
            (sname, h.model.frames[fid].placement.translation))
    axis = {"RX": "1 0 0", "RY": "0 1 0", "RZ": "0 0 1"}

    def body(n, ind):
        s = seg[n]
        if s.axes == ("FIXED",):             # lumped into its parent by Pinocchio
            return []
        j = h.joint_id[n]
        inert = h.model.inertias[j]          # includes rigidly added bodies
        I = inert.inertia
        p = " " * ind
        out = [f'{p}<body name="{n}" pos="{s.offset[0]:.12g} {s.offset[1]:.12g} {s.offset[2]:.12g}">']
        out.append(f'{p} <inertial pos="{inert.lever[0]:.12g} {inert.lever[1]:.12g} {inert.lever[2]:.12g}" '
                   f'mass="{inert.mass:.12g}" fullinertia="{I[0,0]:.12g} {I[1,1]:.12g} {I[2,2]:.12g} '
                   f'{I[0,1]:.12g} {I[0,2]:.12g} {I[1,2]:.12g}"/>')
        for a in s.axes:
            if a == "T":
                for k, ax in enumerate(("1 0 0", "0 1 0", "0 0 1")):
                    out.append(f'{p} <joint name="{s.joint}_{"xyz"[k]}" type="slide" axis="{ax}"/>')
            elif a in ("PZ", "PX"):
                out.append(f'{p} <joint name="{s.joint}_{a[1].lower()}" type="slide" '
                           f'axis="{"0 0 1" if a == "PZ" else "1 0 0"}"/>')
            else:
                out.append(f'{p} <joint name="{s.joint}_{a.lower()}" type="hinge" axis="{axis[a]}" pos="0 0 0"/>')
        for sname, pos in sens.get(n, []):
            out.append(f'{p} <site name="{sname}" pos="{pos[0]:.12g} {pos[1]:.12g} {pos[2]:.12g}"/>')
        for c in kids.get(n, []):
            out += body(c, ind + 1)
        out.append(f"{p}</body>")
        return out

    xml = ['<mujoco model="articulated_human_v1_crosscheck">',
           ' <compiler angle="radian" inertiafromgeom="false" autolimits="false"/>',
           f' <option gravity="0 0 {-h.g:.12g}" timestep="0.001" integrator="RK4">',
           '  <flag contact="disable"/>', ' </option>', ' <worldbody>']
    xml += body("pelvis", 2)
    xml.append(" </worldbody>")
    if sensors:
        xml.append(" <sensor>")
        for sname in h.sensor_frame:
            xml.append(f'  <accelerometer name="acc_{sname}" site="{sname}"/>')
            xml.append(f'  <gyro name="gyr_{sname}" site="{sname}"/>')
        xml.append(" </sensor>")
    xml.append("</mujoco>")
    return "\n".join(xml)
