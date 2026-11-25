"""Mean vertical force under one foot over the ground-contact phase, per gait
case, from the rigid-ground trials (steady part).  In units of local weight.

    python experiments/phase_F_outcomes/gait_profiles.py

A contact starts when the foot's vertical force exceeds 0.02 W and ends when it
falls below again.  Only stances are averaged: contacts shorter than
MIN_STANCE_S are brief touches, nearly all of them toe touches early in the
swing (the toe scuffs of tab_gait), and are left out.  In these trials every
stance lasts 0.55-1.48 s, while every brief touch lasts 0.065 s or less with a
peak of at most 0.31 W, so any limit between the two gives the same result; the
assertions below stop the script if a future gait blurs that separation.  (The
first version kept every contact of at least 10 samples, 0.05 s, which averaged
the toe touches of the side-view walker on the Moon and of the 3-D walker on
Mars with their stances and halved those two curves.)
"""
import glob
import json
import os

import numpy as np

from planetsense.simulation.records import load_record

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REC = os.path.join(ROOT, "outputs", "raw", "records")
ON_W = 0.02             # contact threshold, vertical force in units of local body weight W
MIN_STANCE_S = 0.2      # s, shortest contact counted as a stance (shorter ones are toe touches in the swing)
STEADY_FROM_S = 3.0     # s, contacts starting earlier belong to the start-up and are left out
pct = np.linspace(0.0, 100.0, 101)
out = dict(pct=[float(v) for v in pct],
           note=f"mean over all steady stance phases of both feet, rigid ground (contacts of at least {MIN_STANCE_S:g} s; "
                "brief toe touches in the swing are left out)")
conds = sorted({os.path.basename(p).split("_rigid_")[0][len("walking_"):]
                for p in glob.glob(os.path.join(REC, "walking_*_rigid_r*.npz"))})
assert conds, f"no rigid-ground walking records in {REC}"
for c in conds:
    curves, durations, touches = [], [], []
    planet = None
    for p in sorted(glob.glob(os.path.join(REC, f"walking_{c}_rigid_r*.npz"))):
        r = load_record(p)
        planet = str(r["planet"])
        W = float(r["mass"]) * float(r["g"])
        t = r["t"]
        for j in (0, 1):
            fz = np.asarray(r["fc"][:, j, 2], float) / W
            on = fz > ON_W
            up = np.nonzero(~on[:-1] & on[1:])[0] + 1
            dn = np.nonzero(on[:-1] & ~on[1:])[0] + 1
            for a in up:
                b = dn[dn > a]
                if not len(b) or t[a] < STEADY_FROM_S:
                    continue
                b = b[0]
                if t[b] - t[a] < MIN_STANCE_S:
                    touches.append(float(t[b] - t[a]))
                    continue
                tt = (t[a:b] - t[a]) / (t[b] - t[a]) * 100.0
                curves.append(np.interp(pct, tt, fz[a:b]))
                durations.append(float(t[b] - t[a]))
    assert curves, f"{c}: no stance found"
    # the limit must sit in a wide empty gap between toe touches and stances (factor 2 on either side)
    assert min(durations) > 2 * MIN_STANCE_S, f"{c}: a stance of {min(durations):.3f} s is close to the {MIN_STANCE_S} s limit"
    assert not touches or max(touches) < 0.5 * MIN_STANCE_S, \
        f"{c}: a toe touch of {max(touches):.3f} s is close to the {MIN_STANCE_S} s limit"
    C = np.array(curves)
    out[c] = dict(planet=planet, fz=[float(v) for v in C.mean(axis=0)], fz_sd=[float(v) for v in C.std(axis=0)],
                  n_stances=len(curves), contact_time_s=float(np.mean(durations)), n_toe_touches_left_out=len(touches))
    print(c, planet, "stances", len(curves), "contact time %.3f s" % np.mean(durations), "peak %.2f" % C.mean(axis=0).max(),
          "toe touches left out", len(touches))
json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "gait_profiles.json"), "w"), indent=1)
print("wrote outputs/tables/gait_profiles.json")
