"""Phase O (2026-11-22): the same cases on both structures.

The what-if gaits (the Earth gait tuned for 1.25 m/s, and that gait transferred to Mars and the Moon by dynamic
similarity) had been simulated on the reference bridge only.  This runs them on the habitat floor too, 20
crossings each, with the shakers on (E1) and with footsteps only (E2), with the floor's own walking line and shaker
level, exactly as the floor's other walking trials (experiments/phase_E_records/run_records.py --structure S2).
The records go to outputs/raw/records_S2/ with the floor's other records.  The rigid-ground twins are the reference
bridge's (outputs/raw/records/walking_<cond>_rigid_r<rep>.npz), as for every walking trial on the floor.

Repetition 0 of each case was already computed for the viewer (outputs/raw/viewer_demo_S2/, same code and seeds);
it is copied rather than recomputed.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/phase_O_consistency/whatif_floor_trials.py [--workers 3]
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_E_records"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import run_records as RR  # noqa: E402

WHATIF = ("earth_1.25", "mars_0.77_similar", "moon_0.51_similar")
OUT_S2 = RR.OUT + "_S2"
DEMO = os.path.join(ROOT, "outputs", "raw", "viewer_demo_S2")


def jobs():
    out = []
    for cid, planet, speed, pf, _excs in RR.CONDITIONS:
        if cid not in WHATIF:
            continue
        for r in range(RR.N_REP):
            for exc in ("E1", "E2"):
                out.append(dict(kind="walking", planet=planet, exc=exc, rep=r, cond=cid, speed=speed, pfile=pf,
                                structure="S2", name=f"walking_{cid}_{exc}_r{r}"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT_S2, exist_ok=True)
    for f in sorted(os.listdir(DEMO)) if os.path.isdir(DEMO) else []:
        if f.endswith(".npz") and not os.path.exists(os.path.join(OUT_S2, f)):
            shutil.copy2(os.path.join(DEMO, f), os.path.join(OUT_S2, f))
            print("copied", f, flush=True)
    js = jobs()
    print(len(js), "what-if trials on the habitat floor ->", OUT_S2, flush=True)
    with Pool(a.workers) as pool:
        for name, status, wall in pool.imap_unordered(RR.run, js):
            print(f"{name:42s} {status:10s} {wall:6.0f} s", flush=True)
