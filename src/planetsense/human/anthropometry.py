"""Segment inertial parameters and skeleton geometry for the articulated human.

PROVENANCE AND STATUS
---------------------
* Segment mass fractions, longitudinal centre-of-mass positions and radii of
  gyration: de Leva (1996), "Adjustments to Zatsiorsky-Seluyanov's segment
  inertia parameters", J. Biomech. 29(9):1223-1230, Table 4, males.  The same
  source is reference [51] of the earlier study.
* The digits below are recorded with a ``status`` flag per table.  See
  ``docs/assumptions_and_sources.md`` for the verification state of every
  entry and ``docs/literature/E_human_model_parameters.md`` for the
  transcription cross-check.
* Skeleton geometry that de Leva does not provide (lateral joint-centre
  offsets, foot geometry, sensor mounts) is an ASSUMPTION, listed in
  ``ASSUMED_GEOMETRY`` with a plausible range.  None of it is measured for a
  specific person.  The model is literature-informed, not calibrated.

Segment frames: origin at the proximal joint centre; in the neutral standing
posture every segment frame is aligned with the world frame (x anterior,
y left, z up).  Radii of gyration map to principal moments as
    I_xx = m (r_sag  L)^2   (about the antero-posterior axis)
    I_yy = m (r_tr   L)^2   (about the medio-lateral axis)
    I_zz = m (r_long L)^2   (about the longitudinal axis)
for segments whose long axis is vertical in neutral standing, and with the
longitudinal axis along x for the foot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# de Leva (1996) Table 4, males.  length in mm for the stated endpoints,
# mass in % of body mass, cm in % of length from the first-named endpoint,
# radii of gyration (sagittal, transverse, longitudinal) in % of length.
DE_LEVA_MALE = dict(
    stature_mm=1741.0, mass_kg=73.0,
    status=("verified digit by digit against a scan of Table 4 of the paper and against the "
            "transcription BMClab/BMC @187a48c (0 disagreements); see "
            "docs/literature/E_human_model_parameters.md"),
    segments=dict(
        head=dict(ends="VERT-CERV", length=242.9, mass=6.94, cm=50.02, r=(30.3, 31.5, 26.1)),
        upt=dict(ends="SUPR-XYPH", length=170.7, mass=15.96, cm=29.99, r=(71.6, 45.4, 65.9)),
        mpt=dict(ends="XYPH-OMPH", length=215.5, mass=16.33, cm=45.02, r=(48.2, 38.3, 46.8)),
        lpt=dict(ends="OMPH-MIDH", length=145.7, mass=11.17, cm=61.15, r=(61.5, 55.1, 58.7)),
        upper_arm=dict(ends="SJC-EJC", length=281.7, mass=2.71, cm=57.72, r=(28.5, 26.9, 15.8)),
        forearm=dict(ends="EJC-WJC", length=268.9, mass=1.62, cm=45.74, r=(27.6, 26.5, 12.1)),
        hand=dict(ends="WJC-MET3", length=86.2, mass=0.61, cm=79.00, r=(62.8, 51.3, 40.1)),
        thigh=dict(ends="HJC-KJC", length=422.2, mass=14.16, cm=40.95, r=(32.9, 32.9, 14.9)),
        shank=dict(ends="KJC-AJC", length=440.3, mass=4.33, cm=43.95, r=(25.1, 24.6, 10.2)),
        foot=dict(ends="HEEL-TTIP", length=258.1, mass=1.37, cm=44.15, r=(25.7, 24.5, 12.4)),
    ),
    # longitudinal landmark distances used to place joints (mm)
    landmarks=dict(CERV_MIDH=603.3, SUPR_MIDH=531.9, MIDS_MIDH=515.5),
)

# Geometry not available from de Leva (1996).  ASSUMED nominal values with a
# plausible range; treat as epistemic uncertainty.
ASSUMED_GEOMETRY = dict(
    hip_half_width=dict(value=0.085, range=(0.075, 0.095), unit="m",
                        note="half the distance between hip joint centres"),
    shoulder_half_width=dict(value=0.180, range=(0.160, 0.200), unit="m",
                             note="half the distance between shoulder joint centres"),
    ankle_height=dict(value=0.070, range=(0.060, 0.080), unit="m",
                      note="ankle joint centre above the sole"),
    heel_to_ankle=dict(value=0.060, range=(0.050, 0.070), unit="m",
                       note="horizontal distance from the posterior heel to the ankle joint centre"),
    foot_com_height=dict(value=0.030, range=(0.020, 0.040), unit="m",
                         note="foot centre of mass above the sole"),
    heel_contact_x=dict(value=0.035, range=(0.025, 0.045), unit="m",
                        note="heel contact points, forward of the posterior heel"),
    forefoot_contact_frac=dict(value=0.70, range=(0.65, 0.75), unit="-",
                               note="forefoot contact points as a fraction of foot length from the heel"),
    heel_half_width=dict(value=0.025, range=(0.020, 0.030), unit="m", note=""),
    forefoot_half_width=dict(value=0.040, range=(0.035, 0.050), unit="m", note=""),
)


@dataclass
class Segment:
    name: str
    parent: str | None
    joint: str                 # joint name
    axes: tuple                # sequence of elementary joints, e.g. ("RY","RX","RZ")
    offset: np.ndarray         # joint centre in the parent segment frame (m)
    mass: float                # kg
    com: np.ndarray            # centre of mass in the segment frame (m)
    inertia: np.ndarray        # 3x3 inertia about the COM in segment axes (kg m^2)
    length: float              # reference length (m)


def _diag_inertia(m, L, r):
    return np.diag([m * (r[0] * L / 100.0) ** 2, m * (r[1] * L / 100.0) ** 2,
                    m * (r[2] * L / 100.0) ** 2])


def build_segments(body_mass: float = 75.0, stature: float = 1.75,
                   table: dict = DE_LEVA_MALE, geometry: dict | None = None,
                   mass_scale: dict | None = None) -> dict:
    """Return the 15 segments of the articulated model.

    Segment lengths scale with ``stature / table stature``; masses are the
    tabulated fractions of ``body_mass`` (renormalised so that the 15 segments
    sum to exactly ``body_mass``).  ``mass_scale`` optionally multiplies named
    segment masses (used only by sensitivity studies).
    """
    geo = {k: v["value"] for k, v in ASSUMED_GEOMETRY.items()}
    if geometry:
        geo.update(geometry)
    s = stature / table["stature_mm"]             # table mm -> model m, scaled to stature
    T, LM = table["segments"], table["landmarks"]
    frac_sum = (T["head"]["mass"] + T["upt"]["mass"] + T["mpt"]["mass"] + T["lpt"]["mass"]
                + 2 * sum(T[k]["mass"] for k in ("upper_arm", "forearm", "hand",
                                                 "thigh", "shank", "foot")))
    mk = lambda key: body_mass * T[key]["mass"] / frac_sum
    L = {k: T[k]["length"] * s for k in T}
    seg: dict[str, Segment] = {}

    # ---- pelvis (lower part of trunk); frame origin at the mid-hip point
    m = mk("lpt")
    z_cm = (1.0 - T["lpt"]["cm"] / 100.0) * L["lpt"]          # above MIDH
    seg["pelvis"] = Segment("pelvis", None, "root", ("T", "RZ", "RY", "RX"),
                            np.zeros(3), m, np.array([0, 0, z_cm]),
                            _diag_inertia(m, L["lpt"], T["lpt"]["r"]), L["lpt"])

    # ---- torso = upper + middle part of trunk, origin at OMPH (lumbar joint)
    m_u, m_m = mk("upt"), mk("mpt")
    z_xyph, z_supr = L["mpt"], L["mpt"] + L["upt"]
    z_u = z_supr - T["upt"]["cm"] / 100.0 * L["upt"]
    z_m = z_xyph - T["mpt"]["cm"] / 100.0 * L["mpt"]
    m = m_u + m_m
    z_c = (m_u * z_u + m_m * z_m) / m
    I = (_diag_inertia(m_u, L["upt"], T["upt"]["r"])
         + _diag_inertia(m_m, L["mpt"], T["mpt"]["r"]))
    for mi, zi in ((m_u, z_u), (m_m, z_m)):
        d = zi - z_c
        I[0, 0] += mi * d * d
        I[1, 1] += mi * d * d
    seg["torso"] = Segment("torso", "pelvis", "lumbar", ("RY", "RX", "RZ"),
                           np.array([0, 0, L["lpt"]]), m, np.array([0, 0, z_c]),
                           I, z_supr)

    # ---- head + neck, origin at CERV
    z_cerv = (LM["CERV_MIDH"] * s) - L["lpt"]                 # above OMPH
    m = mk("head")
    seg["head"] = Segment("head", "torso", "neck", ("RY", "RX", "RZ"),
                          np.array([0, 0, z_cerv]), m,
                          np.array([0, 0, (1 - T["head"]["cm"] / 100.0) * L["head"]]),
                          _diag_inertia(m, L["head"], T["head"]["r"]), L["head"])

    z_sh = (LM["MIDS_MIDH"] * s) - L["lpt"]                   # shoulders above OMPH
    for side, sg in (("L", +1.0), ("R", -1.0)):
        # ---- arm chain (hanging, long axis -z)
        m = mk("upper_arm")
        seg[f"upper_arm_{side}"] = Segment(
            f"upper_arm_{side}", "torso", f"shoulder_{side}", ("RY", "RX", "RZ"),
            np.array([0, sg * geo["shoulder_half_width"], z_sh]), m,
            np.array([0, 0, -T["upper_arm"]["cm"] / 100.0 * L["upper_arm"]]),
            _diag_inertia(m, L["upper_arm"], T["upper_arm"]["r"]), L["upper_arm"])
        m = mk("forearm")
        seg[f"forearm_{side}"] = Segment(
            f"forearm_{side}", f"upper_arm_{side}", f"elbow_{side}", ("RY",),
            np.array([0, 0, -L["upper_arm"]]), m,
            np.array([0, 0, -T["forearm"]["cm"] / 100.0 * L["forearm"]]),
            _diag_inertia(m, L["forearm"], T["forearm"]["r"]), L["forearm"])
        m = mk("hand")
        seg[f"hand_{side}"] = Segment(
            f"hand_{side}", f"forearm_{side}", f"wrist_{side}", ("RX", "RY"),
            np.array([0, 0, -L["forearm"]]), m,
            np.array([0, 0, -T["hand"]["cm"] / 100.0 * L["hand"]]),
            _diag_inertia(m, L["hand"], T["hand"]["r"]), L["hand"])
        # ---- leg chain
        m = mk("thigh")
        seg[f"thigh_{side}"] = Segment(
            f"thigh_{side}", "pelvis", f"hip_{side}", ("RY", "RX", "RZ"),
            np.array([0, sg * geo["hip_half_width"], 0]), m,
            np.array([0, 0, -T["thigh"]["cm"] / 100.0 * L["thigh"]]),
            _diag_inertia(m, L["thigh"], T["thigh"]["r"]), L["thigh"])
        m = mk("shank")
        seg[f"shank_{side}"] = Segment(
            f"shank_{side}", f"thigh_{side}", f"knee_{side}", ("RY",),
            np.array([0, 0, -L["thigh"]]), m,
            np.array([0, 0, -T["shank"]["cm"] / 100.0 * L["shank"]]),
            _diag_inertia(m, L["shank"], T["shank"]["r"]), L["shank"])
        # foot: long axis along +x, so the "longitudinal" radius applies to x
        m = mk("foot")
        r = T["foot"]["r"]
        I_foot = np.diag([m * (r[2] * L["foot"] / 100.0) ** 2,
                          m * (r[1] * L["foot"] / 100.0) ** 2,
                          m * (r[0] * L["foot"] / 100.0) ** 2])
        com_x = T["foot"]["cm"] / 100.0 * L["foot"] - geo["heel_to_ankle"]
        seg[f"foot_{side}"] = Segment(
            f"foot_{side}", f"shank_{side}", f"ankle_{side}", ("RY", "RX"),
            np.array([0, 0, -L["shank"]]), m,
            np.array([com_x, 0, -(geo["ankle_height"] - geo["foot_com_height"])]),
            I_foot, L["foot"])
    return seg


def foot_contact_points(seg: dict, geometry: dict | None = None) -> dict:
    """Contact-point coordinates in each foot frame (origin at the ankle).

    Four points per foot (heel medial/lateral, forefoot medial/lateral) on the
    sole plane.  Returned as ``{foot_name: (4, 3) array}``.
    """
    geo = {k: v["value"] for k, v in ASSUMED_GEOMETRY.items()}
    if geometry:
        geo.update(geometry)
    out = {}
    for side in ("L", "R"):
        Lf = seg[f"foot_{side}"].length
        xh = geo["heel_contact_x"] - geo["heel_to_ankle"]
        xf = geo["forefoot_contact_frac"] * Lf - geo["heel_to_ankle"]
        z = -geo["ankle_height"]
        hw, fw = geo["heel_half_width"], geo["forefoot_half_width"]
        out[f"foot_{side}"] = np.array([[xh, +hw, z], [xh, -hw, z],
                                        [xf, +fw, z], [xf, -fw, z]])
    return out


def sensor_mounts(seg: dict) -> dict:
    """Nominal virtual-IMU mounts: ``{name: (segment, position in segment frame)}``.

    ASSUMED locations:
    * wrist_L/R: on the forearm 2 cm proximal of the wrist joint centre and
      3 cm lateral (dorsal side with the palm facing the thigh).
    * pelvis_L/R: on the pelvis at belt height, 8 cm anterior, 12 cm lateral,
      8 cm above the hip joint centres.
    * thigh_L/R: trouser-pocket variant on the anterior proximal thigh.
    Sensor axes coincide with the segment axes.
    """
    out = {}
    for side, sg in (("L", +1.0), ("R", -1.0)):
        Lf = seg[f"forearm_{side}"].length
        out[f"wrist_{side}"] = (f"forearm_{side}", np.array([0.0, sg * 0.03, -(Lf - 0.02)]))
        out[f"pelvis_{side}"] = ("pelvis", np.array([0.08, sg * 0.12, 0.08]))
        out[f"thigh_{side}"] = (f"thigh_{side}", np.array([0.07, sg * 0.02, -0.10]))
    return out
