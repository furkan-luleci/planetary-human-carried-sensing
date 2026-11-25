"""Summary of the trial batch (for the Phase E report): per condition and excitation,
number of trials, falls, simulated duration, time on the deck and the largest
energy-audit error.  Writes outputs/tables/trials_summary.json and prints a
Markdown table.

    python experiments/phase_E_records/summarize_trials.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from planetsense.simulation.records import load_record

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REC = os.path.join(ROOT, "outputs", "raw", "records")


def main():
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(REC, "*.npz"))):
        n = os.path.basename(p)[:-4]
        parts = n.split("_")
        if parts[0] == "standing":
            key = ("standing", parts[1], parts[2])
        else:
            exc = parts[-2]
            cond = "_".join(parts[1:-2])
            key = ("walking", cond, exc)
        groups[key].append(p)
    rows = []
    for (act, cond, exc), ps in sorted(groups.items()):
        rs = [load_record(p) for p in ps]
        dur = [float(r["t"][-1]) for r in rs]
        if act == "walking":
            x = [np.asarray(r["com"])[:, 0] for r in rs]
            on = [float(((xx > 0.0) & (xx < 14.0)).sum() / float(r["fs"])) for xx, r in zip(x, rs)]
        else:
            on = dur
        tl = [r["gait"].get("trunk_lean") for r in rs if act == "walking" and isinstance(r.get("gait"), dict)]
        rows.append(dict(activity=act, condition=cond, excitation=exc, n=len(rs), falls=int(sum(bool(r["fell"]) for r in rs)),
                         duration_s=float(np.mean(dur)), on_deck_s=float(np.mean(on)) if exc != "rigid" else None,
                         energy_drift_max=float(max(float(r["energy_drift_rel"]) for r in rs)),
                         trunk_lean_deg=float(np.degrees(np.mean([t for t in tl if t is not None]))) if any(t is not None for t in tl) else None))
    res = dict(rows=rows, total=sum(r["n"] for r in rows), falls=sum(r["falls"] for r in rows))
    json.dump(res, open(os.path.join(ROOT, "outputs", "tables", "trials_summary.json"), "w"), indent=1)
    print("| Activity | Condition | Excitation | Trials | Falls | Simulated | On deck | Trunk lean | Largest energy error |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['activity']} | {r['condition']} | {r['excitation']} | {r['n']} | {r['falls']} | {r['duration_s']:.1f} s | "
              + ("–" if r["on_deck_s"] is None else f"{r['on_deck_s']:.1f} s") + " | "
              + ("–" if r["trunk_lean_deg"] is None else f"{r['trunk_lean_deg']:.1f}°") + f" | {r['energy_drift_max']:.1e} |")
    print(f"\nTotal trials: {res['total']}; falls: {res['falls']}.")


if __name__ == "__main__":
    main()
