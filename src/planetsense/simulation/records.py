"""Time-domain records: a person standing on, or walking across, the deck.

A record is what one simulated trial delivers: the virtual wearable-sensor
outputs at the device rate, reference accelerations of the deck, the foot
forces, and a down-sampled body and deck motion for drawing.  The simulation
is sampled at 1 kHz and low-pass filtered before decimation to the device rate
(a real inertial sensor has an anti-aliasing filter; plain sub-sampling would
fold heel-strike and high deck modes into the analysis band).
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.signal import decimate

from ..excitation.multisine import make_baseline_shakers, shaker_layout
from ..structure.spine_beam import DeckParams
from .standing import make_standing
from .walking import gait_metrics, make_walker

FS_SIM = 1000.0      # Hz, simulation output
FS_IMU = 200.0       # Hz, device rate after anti-alias filtering
FS_ANIM = 50.0       # Hz, drawing
DT = 2.5e-4          # s, integration step (energy audit: 0.5 ms is too coarse for the stance foot)
DECK_REF_XY = np.array([[3.5, 0.575], [7.0, 0.575], [10.5, 0.575], [7.0, -1.2]])
DECK_REF_NAMES = ["quarter span", "mid span", "three-quarter span", "mid span, far edge"]
DECK_L = DeckParams().L      # S1; records of other structures carry their own length (``deck_L``)


def deck_ref_xy(deck):
    """Reference points on a deck: L/4, L/2 and 3L/4 on its walking line, and mid-span near the far
    edge.  For S1 (line 0.575 m, 3 m wide) these are exactly the points used since Phase E."""
    if "structure" not in deck.meta:          # S1: the very same array as always (bit for bit)
        return DECK_REF_XY
    L, W = float(deck.params.L), float(deck.params.W)
    y = float(deck.meta.get("line_y", 0.575))
    return np.array([[0.25 * L, y], [0.5 * L, y], [0.75 * L, y], [0.5 * L, -0.8 * 0.5 * W]])
WALK_TRIM_S = 1.5    # s discarded after the walker steps onto the deck (the earlier study, Section 2.5)
EDGE_S = 0.25        # s simulated beyond a standing record's kept interval (decimation-filter edge)

KEYPOINTS = ["head_top", "neck", "lumbar", "pelvis",
             "shoulder_L", "elbow_L", "wrist_L", "hand_L", "shoulder_R", "elbow_R", "wrist_R", "hand_R",
             "hip_L", "knee_L", "ankle_L", "heel_L", "toe_L", "hip_R", "knee_R", "ankle_R", "heel_R", "toe_R",
             "imu_wrist_L", "imu_pelvis_L", "imu_thigh_L"]


def keypoints(human, q) -> np.ndarray:
    """World positions (len(KEYPOINTS), 3) of body landmarks for drawing."""
    m, d, seg, jid = human.model, human.data, human.segments, human.joint_id
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    o = lambda n: d.oMi[jid[n]].translation
    at = lambda n, loc: d.oMi[jid[n]].act(np.asarray(loc, float))
    P = [at("head", [0, 0, seg["head"].length]), o("head"), o("torso"), o("pelvis")]
    for s in "LR":
        P += [o(f"upper_arm_{s}"), o(f"forearm_{s}"), o(f"hand_{s}"), at(f"hand_{s}", [0, 0, -seg[f"hand_{s}"].length])]
    for i, s in enumerate("LR"):
        c = human.contact_local[i]
        P += [o(f"thigh_{s}"), o(f"shank_{s}"), o(f"foot_{s}"), at(f"foot_{s}", c[:2].mean(axis=0)),
              at(f"foot_{s}", c[2:].mean(axis=0))]
    for n in ("wrist_L", "pelvis_L", "thigh_L"):
        P.append(d.oMf[human.sensor_frame[n]].translation)
    return np.array(P)


def _dec(x, q):
    """Zero-phase FIR low-pass and decimation along axis 0."""
    return decimate(x, q, ftype="fir", axis=0, zero_phase=True)


def shaker_force(forcing, n_sim):
    """The shaker forces (n_sim samples at FS_SIM, 9 channels), decimated to
    the device rate like the sensor signals.  Used as the reference input for
    force-referenced transfer estimates (it is not given to the blind analysis)."""
    q = int(round(FS_SIM / FS_IMU))
    F = forcing.sample(1.0 / FS_SIM, n_sim, 0.0).T
    n = (n_sim // q) * q
    return _dec(F[:n], q).astype(np.float32)


def _pack(sys_, out, deck, y_line, extra):
    """Turn a simulate() result (1 kHz, record_q=True) into a record dict."""
    h = sys_.h
    qi, qa = int(round(FS_SIM / FS_IMU)), int(round(FS_SIM / FS_ANIM))
    n = (len(out["t"]) // qi) * qi
    sl = slice(0, n)
    rec = dict(t=out["t"][sl][::qi].copy(), fs=FS_IMU, sensors=list(out["sensors"]),
               f=_dec(out["f"][sl], qi).astype(np.float32), w=_dec(out["w"][sl], qi).astype(np.float32),
               fc=_dec(out["fc"][sl], qi).astype(np.float32), com=out["com"][sl][::qi].astype(np.float32),
               g=h.g, mass=h.total_mass, fell=bool(out["fell"]), failure=out.get("failure", ""),
               n_slip=int(out["n_slip"]))
    e = out["energy"]
    through = np.abs(np.diff(e[:, 4])).sum() + np.abs(np.diff(e[:, 5])).sum() + np.abs(np.diff(e[:, 7])).sum() + 1e-9
    rec["energy_drift_rel"] = float(np.abs(e[:, 8] - e[0, 8]).max() / through)
    rec["energy_end"] = e[-1].astype(np.float64)
    rec["energy_cols"] = list(out["energy_cols"])
    # drawing data
    ta = out["t"][sl][::qa]
    rec["t_anim"] = ta.copy()
    rec["kp"] = np.array([keypoints(h, q) for q in out["q"][sl][::qa]], np.float32)
    planar = bool(h.meta.get("sagittal"))
    if extra.get("activity") == "walking" and planar:   # the planar walker lives in y = 0; place it on its line
        rec["kp"][:, :, 1] += y_line
    rec["kp_names"] = list(KEYPOINTS)
    rec["fc_anim"] = out["fc"][sl][::qa].astype(np.float32)
    if deck is not None and sys_.nm:
        eta, etadd = out["eta"][sl], out["etadd"][sl]
        # deck acceleration at fixed reference points and under the person
        rec["a_ref"] = _dec(out["a_surf"][sl], qi).astype(np.float32)
        rec["a_ref_names"] = list(DECK_REF_NAMES)
        xc = out["com"][sl, 0]
        # lateral position of "under the person": the walking line for the planar walker,
        # the centre of mass's own lateral position otherwise
        yc = np.full(len(xc[::qi]), y_line) if planar else out["com"][sl, 1][::qi]
        S = deck.surface_modes(xc[::qi], yc)                                    # (n, 3, nm)
        add = _dec(etadd, qi)
        rec["a_under"] = np.einsum("nam,nm->na", S, add).astype(np.float32)
        rec["on_deck"] = ((xc[::qi] > 0.0) & (xc[::qi] < deck.params.L))
        xs = np.linspace(0.0, deck.params.L, 29)
        Sz = deck.surface_modes(xs, np.full(29, y_line))[:, 2, :]               # (29, nm)
        rec["deck_x"] = xs
        rec["deck_w"] = (eta[::qa] @ Sz.T).astype(np.float32)                   # vertical deflection along the walking line
        rec["deck_freq"] = np.asarray(deck.freq_hz)
        rec["deck_zeta"] = np.asarray(deck.zeta)
        rec["deck_labels"] = list(deck.labels)
        rec["deck_L"] = float(deck.params.L)
        rec["structure"] = str(deck.meta.get("structure", "S1"))
    rec.update(extra)
    return rec


def lead_in_seconds(deck, min_s: float = 20.0, decays: float = 3.0) -> float:
    """the earlier study, Section 2.5: max(20 s, 3 decay times of the first mode)."""
    return float(max(min_s, decays / (float(deck.zeta[0]) * float(deck.omega[0]))))


def walk_record(g, params, leg, deck=None, shaker_rms=0.0, floor_rms=0.0, lead_s=0.0, seed=0, x_start=-3.0,
                x_end=15.5, speed=1.25, y_line=0.575, contact=None, g_gain=None, T_max=None, walker="planar",
                gait_seed=None):
    """One crossing.  ``deck=None`` gives the rigid-floor twin.

    Shakers (the earlier study, Section 2.5): ``shaker_rms`` per force component at
    the three shaker points, plus a vertical broadband floor of ``floor_rms``
    per point; ``shaker_rms`` = 0 with a floor is the earlier study's static-load
    case (footsteps only, apart from the floor).  The force history depends on
    the seed only, not on gravity or on the walker (matched excitation).
    ``lead_s`` > 0 runs the deck alone under the same continuous force
    realization for ``lead_s`` seconds before t = 0, while the walker is still
    on the approach."""
    T = T_max or 1.12 * (x_end - x_start) / speed + 1.0
    T = np.ceil(T)
    forcing, sxy = None, None
    if deck is not None and (shaker_rms > 0 or floor_rms > 0):
        sxy = shaker_layout(deck.params.L, deck.params.W)
        forcing = make_baseline_shakers(T=lead_s + T + 4.0, rms=shaker_rms, floor_rms=floor_rms, seed=seed,
                                    n_points=len(sxy))
    if walker == "3d":
        from .walking3d import make_walker3d
        sys_, y = make_walker3d(g, params, leg, deck=deck, contact=contact, x0=x_start, v0=speed,
                                y_walk=y_line, shaker_xy=sxy, g_gain=g_gain, seed=gait_seed)
    else:
        sys_, y = make_walker(g, params, leg, deck=deck, contact=contact, x0=x_start, v0=speed,
                              y_walk=y_line, shaker_xy=sxy, g_gain=g_gain)
    if forcing is not None and lead_s > 0:
        eta0, etad0 = sys_.deck_lead_in(forcing, lead_s, DT)
        ie = 2 * sys_.nq
        y[ie:ie + sys_.nm], y[ie + sys_.nm:ie + 2 * sys_.nm] = eta0, etad0
    out = sys_.simulate(y, T=float(T), dt=DT, out_dt=1.0 / FS_SIM, forcing=forcing, fall_height=0.6,
                        record_q=True, surface_out_xy=deck_ref_xy(deck) if deck is not None else None)
    gm = gait_metrics(out, sys_.h, t_skip=min(3.0, 0.3 * T), i_act=sys_.controller.i_act)
    if walker == "3d" and not out["fell"]:
        from .walking3d import lateral_metrics
        gm.update(lateral_metrics(out, sys_.h, sys_.controller.events, t_skip=min(4.0, 0.3 * T),
                                  step_frequency=gm.get("step_frequency")))
    ev = np.array([(e[0], 0 if e[1] == "L" else 1, e[2]) for e in sys_.controller.events]).reshape(-1, 3)
    rec = _pack(sys_, out, deck, y_line, dict(activity="walking", events=ev, x_start=x_start, x_end=x_end,
                                              target_speed=speed, shaker_rms=shaker_rms, floor_rms=floor_rms,
                                              lead_s=float(lead_s) if forcing is not None else 0.0,
                                              seed=seed, T_sim=float(T), walker=walker))
    if forcing is not None:
        rec["F_shaker"] = shaker_force(forcing, len(out["t"]))
    rec["gait"] = {k: v for k, v in gm.items()}
    return rec


def stand_record(g, leg, deck, shaker_rms=0.0, floor_rms=0.0, lead_s=0.0, seed=0, T=60.0, x_com=7.0, y_com=0.575,
                 **kw):
    """A person standing on the deck.  ``lead_s`` > 0 prepends a lead-in of the
    COUPLED person-deck system (the earlier study, Section 2.5); it is kept in the
    record and marked by ``t_analysis_start``."""
    sc = make_standing(g=g, deck=deck, leg=leg, x_com=x_com, y_com=y_com, **kw)
    sys_ = sc.system
    # 0.25 s beyond the kept interval: the zero-phase decimation filter tapers the signals (gravity
    # included) over its half-length at both ends of a record, so the analysed interval stays clear of it
    T_all = float(lead_s) + float(T) + EDGE_S
    forcing = (make_baseline_shakers(T=T_all + 4.0, rms=shaker_rms, floor_rms=floor_rms, seed=seed)
               if ((shaker_rms > 0 or floor_rms > 0) and deck is not None) else None)
    out = sys_.simulate(sc.y_eq.copy(), T=T_all, dt=DT, out_dt=1.0 / FS_SIM, forcing=forcing, fall_height=0.7,
                        record_q=True, surface_out_xy=deck_ref_xy(deck) if deck is not None else None)
    W = sc.info["weight"]
    # bridge motion "under the person" is read on the line through the feet (y_com), not on the centreline
    rec = _pack(sys_, out, deck, y_com, dict(activity="standing", shaker_rms=shaker_rms, floor_rms=floor_rms,
                                           seed=seed, x_com=x_com, y_com=y_com, T_sim=T_all,
                                           lead_s=float(lead_s), t_analysis_start=float(lead_s),
                                           t_analysis_end=float(lead_s) + float(T)))
    if forcing is not None:
        rec["F_shaker"] = shaker_force(forcing, len(out["t"]))
    m = (out["t"] >= lead_s) & (out["t"] < lead_s + T)
    fz = out["fn"][m].sum(axis=1) / W
    nopen = (out["fn"][m] <= 0).sum(axis=1)
    rec["contact"] = dict(sumFn_over_W_min=float(fz.min()), sumFn_over_W_max=float(fz.max()),
                          frac_time_any_point_open=float((nopen > 0).mean()))
    return rec


def analysis_window(rec, trim_s: float = WALK_TRIM_S):
    """Samples analysed from a record (boolean mask on ``rec["t"]``).

    Walking (the earlier study, Section 2.5): the crossing, from the moment the
    centre of mass passes the near support (x = 0) to the moment it passes the
    far one (x = L), minus the first ``trim_s`` seconds.  Standing: everything
    after the coupled lead-in up to the end of the kept interval (records made
    before lead-ins existed: t >= 4 s).
    """
    t = np.asarray(rec["t"], float)
    if rec["activity"] == "standing":
        m = t >= float(rec.get("t_analysis_start", 4.0) or 4.0)
        if rec.get("t_analysis_end") is not None:
            m &= t < float(rec["t_analysis_end"])
        return m
    x = np.asarray(rec["com"], float)[:, 0]
    on = np.nonzero(x > 0.0)[0]
    off = np.nonzero(x >= float(rec.get("deck_L", DECK_L)))[0]
    if not on.size:
        return np.zeros(len(t), bool)
    t_in = t[on[0]]
    t_out = t[off[0]] if off.size else t[-1] + 1.0
    return (t >= t_in + trim_s) & (t < t_out)


def crossing_window(rec):
    """Walking: the whole crossing (x from 0 to L), without the start-up trim."""
    return analysis_window(rec, trim_s=0.0)


def gait_clock(rec, fp):
    """Gait clock tau(t) (s) of a walking record, for order tracking of the
    gait harmonics as in the earlier study (Eq. 6): the step phase (touchdown
    events, alternating feet) divided by the nominal step frequency ``fp``,
    so that gait harmonics sit at exact multiples of fp/2 on tau.  Between
    touchdowns the phase grows linearly; before the first and after the last
    it is extrapolated with the neighbouring step duration."""
    t = np.asarray(rec["t"], float)
    ev = np.asarray(rec["events"], float).reshape(-1, 3)
    te = ev[:, 0]
    if te.size < 3 or np.any(np.diff(te) <= 0) or np.any(np.diff(ev[:, 1]) == 0):
        raise ValueError("gait_clock: touchdowns must alternate feet and increase in time")
    j = np.arange(te.size, dtype=float)
    psi = np.interp(t, te, j)
    lo, hi = t < te[0], t > te[-1]
    psi[lo] = (t[lo] - te[0]) / (te[1] - te[0])
    psi[hi] = j[-1] + (t[hi] - te[-1]) / (te[-1] - te[-2])
    return psi / float(fp)


def save_record(path, rec):
    flat = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            import json
            flat["json_" + k] = json.dumps(v, default=float)
        else:
            flat[k] = v
    np.savez_compressed(path, **flat)


def load_record(path):
    import json
    z = np.load(path, allow_pickle=False)
    rec = {}
    for k in z.files:
        if k.startswith("json_"):
            rec[k[5:]] = json.loads(str(z[k]))
        else:
            v = z[k]
            rec[k] = v.item() if v.shape == () else v
    return rec
