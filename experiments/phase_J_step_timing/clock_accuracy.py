"""Phase J: accuracy of the device-estimated gait clocks against the touchdown clock,
for every walking case used by the step-timing test (records and their rigid-ground
twins, each device position), with the pooling of step_timing_test.py.  No spectra,
no index.

Writes outputs/tables/step_timing_clocks.json.

    python experiments/phase_J_step_timing/clock_accuracy.py [--workers 4]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step_timing_test as STT  # noqa: E402

ROOT = STT.ROOT


def one(tk):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from planetsense.simulation.records import load_record
    recs = [load_record(f) for f in tk["files"]]
    twins = [load_record(f) for f in tk["twins"]]
    keep = [i for i in range(len(recs)) if not recs[i].get("fell") and not twins[i].get("fell")]
    c = STT.device_clocks([recs[i] for i in keep], [twins[i] for i in keep])
    return dict(sid=tk["sid"], planet=tk["planet"], cond=tk["cond"],
                sensors={s: {k: v for k, v in d.items() if k not in ("clocks_rec", "clocks_twin")} for s, d in c.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = list(pool.imap(one, STT.walking_tasks()))
    rows = []
    for r in res:
        for s, d in r["sensors"].items():
            A = d["accuracy"]
            tot = np.array([x["total_ms"] for x in A]); rnd = np.array([x["random_ms"] for x in A])
            mx = np.array([x["max_ms"] for x in A]); T = np.median([x["true_stride_s"] for x in A])
            rows.append(dict(sid=r["sid"], planet=r["planet"], cond=r["cond"], sensor=s, n_e=d["n_e"],
                             stride_s=d["stride_s"], stride_true_s=float(T),
                             stride_error_pct=float(100 * (d["stride_s"] - T) / T),
                             total_ms_median=float(np.median(tot)), total_ms_p95=float(np.percentile(tot, 95)),
                             random_ms_median=float(np.median(rnd)), random_ms_p95=float(np.percentile(rnd, 95)),
                             random_ms_max=float(rnd.max()), max_abs_ms=float(mx.max()),
                             records_over_30ms=int((tot > 30).sum()), records=len(A)))
    allr = [x for r in res for d in r["sensors"].values() for x in d["accuracy"]]
    tot = np.array([x["total_ms"] for x in allr]); rnd = np.array([x["random_ms"] for x in allr])
    summ = dict(clock_estimates=len(allr), total_ms_median=float(np.median(tot)), total_ms_p95=float(np.percentile(tot, 95)),
                random_ms_median=float(np.median(rnd)), random_ms_p95=float(np.percentile(rnd, 95)),
                over_30ms=int((tot > 30).sum()),
                wrong_events_per_stride=int(sum(abs(x['stride_error_pct']) > 10 for x in rows)),
                case_positions=len(rows), max_abs_stride_error_pct=float(max(abs(x['stride_error_pct']) for x in rows)))
    json.dump(dict(rows=rows, summary=summ, cases=res, wall_seconds=time.time() - t0),
              open(os.path.join(ROOT, "outputs", "tables", "step_timing_clocks.json"), "w"), indent=1, default=float)
    for x in rows:
        print(f"{x['sid']:16s} {x['planet']:5s} {x['sensor']:8s} n_e {x['n_e']} stride {x['stride_s']:.3f} ({x['stride_error_pct']:+.1f} %) "
              f"| total med {x['total_ms_median']:5.1f} p95 {x['total_ms_p95']:6.1f} | random med {x['random_ms_median']:5.1f} "
              f"p95 {x['random_ms_p95']:5.1f} max {x['random_ms_max']:6.1f} | >30ms {x['records_over_30ms']}/{x['records']}")
    print("summary", summ, f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
