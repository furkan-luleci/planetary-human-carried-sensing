"""Generate the time-domain trials (records) of the study.

Settings follow the earlier study (Sections 2.5 and Table 1) wherever the two
studies overlap (2026-11-19, "baseline-matched" set):

  shakers       three points (L/4, L/2, 3L/4 on a diagonal), 800 N RMS per force
                component (x, y, z) plus a 5 N vertical broadband floor per point
  lead-in       walking: the deck alone under the same continuous shaker forces
                for max(20 s, 3 decay times of B1) = 20 s before the record;
                standing: the coupled person-deck system for 20 s, then 60 s kept
  records       20 per case (walking crossings; standing records)

Trials
  standing at 4.2 m, shakers on (E1)                           3 planets x N_STAND
  standing at mid-span, Earth, shakers on (earlier-study baseline)      1 x N_STAND
  walking across the deck, shakers on (E1, matched forces)     each gait condition x N_REP
  walking across the deck, footsteps only (E2; the 5 N floor
      is the only shaker force: the earlier study's static-load case)       each gait condition x N_REP
  walking the same line on rigid ground (twin)                 each gait condition x N_REP

"Matched" means the shaker force history depends only on the repetition seed:
the same forces are applied on every planet.  Repetitions differ in the
starting position of the walker (shifted by a twentieth of a step from one
repetition to the next, which moves every footfall along the deck) and in the
force seed.

    python experiments/phase_E_records/run_records.py [--only standing|walking] [--workers 4] [--structure S1|S2|S2L]

Phase K (2026-11-21): ``--structure S2`` or ``S2L`` runs the second structure (src/planetsense/structure/registry.py)
into outputs/raw/records_<name>/: standing on three planets and the six main gaits (1.0 m/s, side-view and
3-D walkers) with shakers (E1) and footsteps only (E2), with the structure's own standing point, walking line
and shaker level (rules declared in docs/phase_reports/PHASE_K_declared_rules.md).  No rigid-ground twins are
run: S1's are reused (rigid ground does not see the structure; the starts and seeds are the same).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs", "raw", "records")
GAIT = os.path.join(ROOT, "outputs", "gait")
G = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}
SHAKER_RMS = 800.0          # N per force component (the earlier study, Table 1)
FLOOR_RMS = 5.0             # N, vertical broadband floor per shaker point (the earlier study, Section 2.5)
X_STAND = 4.2               # m; this study's standing position (no vertical bending mode below 30 Hz has a node there)
X_STAND_MID = 7.0           # m; mid-span, the earlier-study baseline standing position (Earth check)
N_STAND, N_REP = 20, 20
# gait conditions: (id, planet, target speed, parameter file, excitations)
ALL_EXC = ("E1", "E2", "rigid")
CONDITIONS = [
    ("earth_1.25", "Earth", 1.25, "earth_1.25.json", ALL_EXC),
    ("earth_1.00", "Earth", 1.00, "earth_1.00.json", ALL_EXC),
    ("mars_1.00", "Mars", 1.00, "mars_1.00.json", ALL_EXC),
    ("moon_1.00", "Moon", 1.00, "moon_1.00.json", ALL_EXC),
    ("mars_0.77_similar", "Mars", 0.77, "mars_0.77_similar.json", ALL_EXC),
    ("moon_0.51_similar", "Moon", 0.51, "moon_0.51_similar.json", ALL_EXC),
    ("earth3d_1.00", "Earth", 1.00, "earth3d_1.00.json", ALL_EXC),       # Phase G: 3-D walker, Stage 1
    ("mars3d_1.00", "Mars", 1.00, "mars3d_1.00.json", ALL_EXC),          # Phase G, Stage 2 (2026-11-20)
    ("moon3d_1.00", "Moon", 1.00, "moon3d_1.00.json", ALL_EXC),
    ("earth_1.50", "Earth", 1.50, "earth_1.50.json", ("E1", "rigid")),  # earlier-study baseline walking speed (Earth check)
]
# conditions whose walker starts at the speed its gait was tuned from instead of its measured mean speed:
# the 1.5 m/s gait falls within 4 s when started at its mean speed (1.46 m/s) but walks when started at 1.5 m/s
START_AT_TUNING_SPEED = {"earth_1.50"}


MAIN_S2 = ("earth_1.00", "mars_1.00", "moon_1.00", "earth3d_1.00", "mars3d_1.00", "moon3d_1.00")


def jobs(only=None, structure="S1"):
    """Walking first (shorter trials; the analysis can start on them), then standing."""
    js = []
    if structure != "S1":
        if only in (None, "walking"):
            for cid, planet, speed, pf, _excs in CONDITIONS:
                if cid not in MAIN_S2:
                    continue
                for r in range(N_REP):
                    for exc in ("E1", "E2"):
                        js.append(dict(kind="walking", planet=planet, exc=exc, rep=r, cond=cid, speed=speed, pfile=pf,
                                       structure=structure, name=f"walking_{cid}_{exc}_r{r}"))
        if only in (None, "standing"):
            for r in range(N_STAND):
                for planet in G:
                    js.append(dict(kind="standing", planet=planet, exc="E1", rep=r, structure=structure,
                                   name=f"standing_{planet}_E1_r{r}"))
        return js
    if only in (None, "walking"):
        for cid, planet, speed, pf, excs in CONDITIONS:
            if not os.path.exists(os.path.join(GAIT, pf)):
                continue
            for r in range(N_REP):
                for exc in excs:
                    js.append(dict(kind="walking", planet=planet, exc=exc, rep=r, cond=cid, speed=speed,
                                   pfile=pf, name=f"walking_{cid}_{exc}_r{r}"))
    if only in (None, "standing"):
        for r in range(N_STAND):
            for planet in G:
                js.append(dict(kind="standing", planet=planet, exc="E1", rep=r, x=X_STAND,
                               name=f"standing_{planet}_E1_r{r}"))
            js.append(dict(kind="standing", planet="Earth", exc="E1", rep=r, x=X_STAND_MID,
                           name=f"standing_Earth-mid_E1_r{r}"))
    return js


def run(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.human.body_v2 import load_leg
    from planetsense.simulation.records import lead_in_seconds, save_record, stand_record, walk_record
    from planetsense.simulation.walking import WalkParams
    from planetsense.structure import registry as SR
    st = SR.get(job.get("structure", "S1"))
    out_dir = OUT if st.name == "S1" else OUT + "_" + st.name
    path = os.path.join(out_dir, job["name"] + ".npz")
    if os.path.exists(path):
        return job["name"], "exists", 0.0
    t0 = time.perf_counter()
    g = G[job["planet"]]
    deck = SR.modal_deck(st.name)
    lead = lead_in_seconds(deck)
    leg = load_leg()
    if job["kind"] == "standing":
        if st.name == "S1":
            rec = stand_record(g, leg, deck, shaker_rms=SHAKER_RMS, floor_rms=FLOOR_RMS, lead_s=lead,
                               seed=100 + job["rep"], T=60.0, x_com=job["x"])
        else:
            rec = stand_record(g, leg, deck, shaker_rms=st.shaker_rms, floor_rms=st.floor_rms, lead_s=lead,
                               seed=100 + job["rep"], T=60.0, x_com=st.stand_xy[0], y_com=st.stand_xy[1])
    else:
        pj = json.load(open(os.path.join(GAIT, job["pfile"])))
        walker, gseed = pj.get("walker", "planar"), None
        if walker == "3d":
            from planetsense.simulation.walking3d import WalkParams3D, planar_seed
            p = WalkParams3D(**pj["params"])
            gseed = planar_seed(g, None, leg, v0=job["speed"], path=os.path.join(ROOT, pj["seed_from"]))
        else:
            p = WalkParams(**pj["params"])
        mv = pj.get("metrics_validation_30s", {})
        speed = mv.get("speed", job["speed"])
        step = speed / mv["step_frequency"] if mv.get("step_frequency") else 0.55
        # approach of at least 4 m; the start moves by a twentieth of a step per repetition
        x0 = -max(4.0, 5.0 * speed) - step * job["rep"] / N_REP
        v_start = float(pj["target_speed"]) if job["cond"] in START_AT_TUNING_SPEED else speed
        sh, fl = (SHAKER_RMS, FLOOR_RMS) if st.name == "S1" else (st.shaker_rms, st.floor_rms)
        extra = {} if st.name == "S1" else dict(x_end=float(deck.params.L) + 1.5, y_line=st.line_y)
        rec = walk_record(g, p, leg, deck=None if job["exc"] == "rigid" else deck,
                          shaker_rms=sh if job["exc"] == "E1" else 0.0,
                          floor_rms=fl if job["exc"] in ("E1", "E2") else 0.0, lead_s=lead,
                          seed=100 + job["rep"], x_start=x0, **dict(dict(x_end=15.5), **extra), speed=v_start,
                          g_gain=pj.get("g_gain"), walker=walker, gait_seed=gseed)
        rec["condition"] = job["cond"]
    rec.update(planet=job["planet"], excitation=job["exc"], rep=job["rep"], name=job["name"])
    os.makedirs(out_dir, exist_ok=True)
    save_record(path, rec)
    return job["name"], ("FELL " + str(rec["failure"])) if rec["fell"] else "ok", time.perf_counter() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--structure", default="S1", choices=("S1", "S2", "S2L"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    js = jobs(a.only, a.structure)
    print(len(js), "jobs", flush=True)
    with Pool(a.workers) as pool:
        for name, status, wall in pool.imap_unordered(run, js):
            print(f"{name:42s} {status:10s} {wall:6.0f} s", flush=True)
