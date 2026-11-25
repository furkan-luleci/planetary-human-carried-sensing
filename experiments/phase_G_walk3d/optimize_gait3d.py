"""Tune the 3-D walking controller (Phase G, Stage 1: Earth) with CMA-ES.

Same procedure as experiments/phase_C_walking/optimize_gait.py (declared
targets, rejected falls and failed energy audits, checkpoint per generation,
30 s stability selection of the 16 best candidates), with the 3-D walker of
planetsense.simulation.walking3d and frontal-plane targets added.  Each evaluation
starts from a state on the planar walker's steady gait cycle (outputs/gait/
_seed_planar_<case>.npz) with the start-up aid of WalkController3D active for
the first 2-3 s; descriptors are taken after 4 s.

ORIGINAL DESCRIPTION (planar tuner, kept for reference):
Tune the walking controller for one gravity and one target speed (CMA-ES).

The controller (planetsense.simulation.walking) is a control hypothesis with 14 free
parameters.  This script finds parameters for which the forward simulation
walks steadily at a prescribed speed and matches a set of gait descriptors
(``--targets``).  What is matched is DECLARED, not hidden:

  Earth          targets are published human values (calibration, not prediction)
  Mars / Moon    targets are limited to what reduced-gravity experiments report
                 (speed, step frequency, duty factor); everything else,
                 including ground forces, is an outcome of the simulation

Parameters are searched in Earth-equivalent units and mapped to the local
gravity with ``WalkParams.similarity_scaled`` (times stretch by sqrt(g_E/g),
joint gains scale with g), so the same bounds apply on every planet.

Every evaluation is a forward simulation from a standing start moving at the
target speed; a fall is a (large) cost, not an exception.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from planetsense.human.body_v2 import load_leg
from planetsense.simulation.walking import WalkParams, gait_metrics
from planetsense.simulation.walking3d import WalkParams3D, lateral_metrics, make_walker3d, planar_seed

FREE = ["T_ds", "T0", "hip0", "hip1", "knee0", "st_knee", "k_knee", "st_ankle", "k_ank",
        "foot_pitch", "lean", "c_d", "c_v", "push", "po_ankle", "gain",
        "w0", "cd_lat", "cv_lat", "k_ank_lat", "k_hip_lat", "zab", "lam_lat", "k_yaw", "brake", "lumbar_ext"]
# design choices held fixed during tuning (round c): heading for foot placement = pelvis yaw
# low-pass filtered with a 1 s time constant (about one stride)
FIXED = dict(tau_head=1.0)
LO = np.array([0.04, 0.06, 0.10, -0.30, 0.5, 0.00, 0.4, -0.15, 0.8, -0.30, -0.02, 0.0, 0.0, 0.0, 0.0, 0.15,
               -0.15, 0.0, 0.0, 0.3, 0.3, 0.7, 0.0, 0.5, 0.0, -0.05])
HI = np.array([0.20, 0.30, 0.70, 0.50, 1.3, 0.30, 2.0, 0.10, 5.0, 0.05, 0.20, 2.5, 0.5, 60.0, 0.6, 1.50,
               0.05, 1.5, 0.6, 4.0, 2.0, 3.0, 1.0, 4.0, 600.0, 0.30])
G_E = 9.81
DT = 2.5e-4          # s; 0.5 ms is beyond the stability limit of the stance foot (energy audit)
DRIFT_MAX = 1e-3     # a candidate whose energy audit fails is rejected as numerically invalid

# default descriptor targets: (value, tolerance, kind); kind "eq" = match, "max" = upper bound
EARTH_TARGETS = {
    1.0: dict(speed=(1.0, 0.04, "eq"), step_frequency=(1.67, 0.08, "eq"), duty_factor=(0.64, 0.02, "eq"),
              dlf1=(0.27, 0.06, "eq"), grf_foot_peak_over_W=(1.12, 0.08, "max"), grf_peak_over_W=(1.45, 0.15, "max"),
              step_width=(0.10, 0.03, "eq"), dlf_lat=(0.04, 0.02, "eq"), grf_lateral_peak_over_W=(0.15, 0.05, "max"),
              lateral_drift=(0.15, 0.10, "max"), heading_max=(0.15, 0.10, "max"), swing_scuffs_per_step=(0.0, 0.25, "eq"),
              trunk_lean=(0.07, 0.025, "max")),
}
COMMON = dict(step_time_cv=(0.0, 0.03, "eq"), flight_fraction=(0.0, 0.01, "eq"), effort_abs=(0.0, 0.03, "eq"))
SEEDS = {1.0: ("outputs/gait/earth_1.00.json", "outputs/gait/_seed_planar_earth_1.00.npz")}

_CFG = {}


def to_params(x01, g):
    x = LO + np.clip(x01, 0.0, 1.0) * (HI - LO)
    return WalkParams3D(**{k: float(v) for k, v in zip(FREE, x)}, **FIXED)


def _seed():
    if "seed" not in _CFG:
        pj_path, seed_path = _CFG.get("seed_files") or SEEDS[round(_CFG["v0"], 2)]
        pj = json.load(open(pj_path))
        _CFG["seed"] = planar_seed(_CFG["g"], WalkParams(**pj["params"]), load_leg(_CFG.get("leg", "nominal_n1.5_constant")),
                                   v0=_CFG["v0"], path=seed_path, g_gain=pj.get("g_gain"))
    return _CFG["seed"]


def simulate(x01, g, v0, T, t_skip, leg=None, record_q=True):
    p = to_params(x01, g)
    sys_, y = make_walker3d(g, p, leg or load_leg(_CFG.get("leg", "nominal_n1.5_constant")), v0=v0, y_walk=0.0,
                            g_gain=_CFG.get("g_gain"), seed=_seed())
    out = sys_.simulate(y, T=T, dt=DT, out_dt=5e-3, fall_height=0.6, record_q=record_q)
    m = gait_metrics(out, sys_.h, t_skip=t_skip, i_act=sys_.controller.i_act)
    if "dlf" in m:
        m["dlf1"] = m["dlf"][0]
    if not out["fell"]:
        m.update(lateral_metrics(out, sys_.h, sys_.controller.events, t_skip=t_skip, step_frequency=m.get("step_frequency")))
    return out, m, sys_


def cost(x01):
    c = _CFG
    try:
        out, m, _ = simulate(x01, c["g"], c["v0"], c["T"], c["t_skip"])
    except Exception as e:                       # numerical failure counts as a fall
        return 2000.0, dict(error=str(e)[:80])
    if out["fell"]:
        return 1000.0 + 200.0 * (c["T"] - out["t_end"]) / c["T"], m
    if "step_frequency" not in m or not m.get("alternating", False):
        return 600.0, m
    if m["energy_drift_rel"] > DRIFT_MAX:
        return 800.0, m
    J = 0.0
    for k, (val, tol, kind) in c["targets"].items():
        if k not in m:
            J += 25.0
            continue
        e = (m[k] - val) / tol
        if kind == "max":
            e = max(e, 0.0)
        J += e * e
    return float(J), m


def validate(x01):
    """30 s run on rigid ground with the candidate (for the final selection)."""
    out, m, _ = simulate(x01, _CFG["g"], _CFG["v0"], 30.0, 8.0)
    return bool(out["fell"]), m


def _init(cfg):
    _CFG.update(cfg)
    if cfg.get("hi") is not None:          # per-run bounds (--hi, --lo) reach the worker processes
        HI[:] = np.asarray(cfg["hi"], float)
    if cfg.get("lo") is not None:
        LO[:] = np.asarray(cfg["lo"], float)
    np.seterr(all="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=G_E)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--targets", type=str, default=None, help="JSON dict name -> [value, tol, kind]")
    ap.add_argument("--x0", type=str, default=None, help="result JSON to start from")
    ap.add_argument("--random", type=int, default=240)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=float, default=9.0)
    ap.add_argument("--validate-top", type=int, default=16)
    ap.add_argument("--init", type=str, default=None, help="JSON dict of starting values for parameters not taken from --x0")
    ap.add_argument("--t-skip", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--leg", type=str, default="nominal_n1.5_constant")
    ap.add_argument("--no-similarity", action="store_true",
                    help="search in physical units (no time stretch); use with --g-gain 9.81 for re-tuned gaits")
    ap.add_argument("--g-gain", type=float, default=None, help="gravity the joint gains are scaled for (default: local g)")
    ap.add_argument("--hi", type=str, default=None,
                    help="JSON dict of upper bounds that replace the defaults for this run (the --x0 file's x01 must use them)")
    ap.add_argument("--lo", type=str, default=None, help="JSON dict of lower bounds that replace the defaults for this run")
    ap.add_argument("--seed-gait", type=str, default=None,
                    help="planar gait JSON whose steady cycle starts each evaluation (default: the Earth gait of --speed)")
    ap.add_argument("--seed-npz", type=str, default=None, help="cache file of that seed state")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    import cma
    for k, v in (json.loads(a.hi).items() if a.hi else []):
        HI[FREE.index(k)] = float(v)
    for k, v in (json.loads(a.lo).items() if a.lo else []):
        LO[FREE.index(k)] = float(v)

    if a.targets:
        tg = {k: tuple(v) for k, v in json.loads(a.targets).items()}
    else:
        tg = dict(EARTH_TARGETS[round(a.speed, 2)])
    tg = {**COMMON, **tg}
    seed_files = (a.seed_gait, a.seed_npz) if a.seed_gait else SEEDS[round(a.speed, 2)]
    cfg = dict(g=a.g, v0=a.speed, T=a.T, t_skip=a.t_skip, targets=tg, leg=a.leg,
               similarity=not a.no_similarity, g_gain=a.g_gain, seed_files=seed_files, hi=HI.tolist(), lo=LO.tolist())
    _init(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = open(a.out.replace(".json", ".log"), "a")
    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    hist = []
    ckpt = a.out.replace(".json", ".ckpt.json")
    resumed = None
    if os.path.exists(ckpt):
        resumed = json.load(open(ckpt))
        hist = resumed["history"]
        print(f"resuming from checkpoint after generation {len(hist)} (best {resumed['cost']:.2f})", file=log, flush=True)
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        if resumed is not None:
            x0, J0 = np.array(resumed["x01"]), float(resumed["cost"])
            a.sigma = float(resumed["sigma"])
            a.maxiter = max(1, a.maxiter - len(hist))
        elif a.x0:
            j0 = json.load(open(a.x0))
            x0 = np.full(len(FREE), 0.0)
            for k, v in zip(j0["free"], j0["x01"]):          # parameters added since are started at their default
                x0[FREE.index(k)] = v
            init = WalkParams3D(**json.loads(a.init)) if a.init else WalkParams3D()
            for k in FREE:
                if k not in j0["free"] or (a.init and k in json.loads(a.init)):
                    x0[FREE.index(k)] = np.clip((getattr(init, k) - LO[FREE.index(k)]) / (HI[FREE.index(k)] - LO[FREE.index(k)]), 0, 1)
            X = [x0] + [np.clip(x0 + a.sigma * rng.standard_normal(len(FREE)), 0, 1) for _ in range(max(0, a.random - 1))]
            R = pool.map(cost, X, chunksize=2)
            Js = np.array([r[0] for r in R])
            i = int(np.argmin(Js))
            x0, J0 = np.array(X[i]), Js[i]
            print(f"warm start from {a.x0}: start cost {Js[0]:.2f}, best of {len(X)} neighbours {J0:.2f}; "
                  f"walking {int((Js < 600).sum())}/{len(Js)}", file=log, flush=True)
        else:
            X = rng.uniform(0.0, 1.0, size=(a.random, len(FREE)))
            R = pool.map(cost, list(X), chunksize=4)
            Js = np.array([r[0] for r in R])
            i = int(np.argmin(Js))
            x0, J0 = X[i], Js[i]
            print(f"random stage: best {J0:.2f}; walking {int((Js < 600).sum())}/{len(Js)}; "
                  f"{time.time() - t_start:.0f} s", file=log, flush=True)
        es = cma.CMAEvolutionStrategy(list(x0), a.sigma, dict(bounds=[0.0, 1.0], popsize=a.popsize,
                                                               seed=a.seed, verbose=-9, maxiter=a.maxiter))
        best = (J0, np.array(x0), {})
        cand = [(float(J0), np.array(x0))]
        while not es.stop():
            if os.path.exists(a.out + ".stop"):          # manual early stop: go straight to the final selection
                print(f"stopped early after generation {len(hist)} (stop file)", file=log, flush=True)
                break
            X = es.ask()
            R = pool.map(cost, X, chunksize=1)
            Js = [r[0] for r in R]
            es.tell(X, Js)
            cand += [(float(J), np.array(x)) for J, x in zip(Js, X) if J < 600.0]
            i = int(np.argmin(Js))
            if Js[i] < best[0]:
                best = (Js[i], np.array(X[i]), R[i][1])
            hist.append(dict(gen=len(hist) + 1, best=float(best[0]), gen_best=float(Js[i]),
                             median=float(np.median(Js)), sigma=float(es.sigma)))
            json.dump(dict(x01=[float(v) for v in best[1]], cost=float(best[0]), sigma=float(es.sigma),
                           metrics=best[2], history=hist, free=FREE), open(ckpt, "w"))
            mm = best[2]
            print(f"gen {len(hist):3d} best {best[0]:8.2f} gen {Js[i]:8.2f} med {np.median(Js):8.1f} "
                  f"sig {es.sigma:.3f} | v {mm.get('speed', 0):.2f} f {mm.get('step_frequency', 0):.2f} "
                  f"df {mm.get('duty_factor', 0):.2f} dlf1 {mm.get('dlf1', 0):.2f} "
                  f"pk {mm.get('grf_foot_peak_over_W', 0):.2f}/{mm.get('grf_peak_over_W', 0):.2f} "
                  f"eff {mm.get('effort', 0):.3f} | {time.time() - t_start:.0f} s", file=log, flush=True)
        # final selection: the best K distinct candidates are run for 30 s; the best
        # (6-s cost) candidate that does not fall and still walks alternately is kept
        cand.sort(key=lambda c: c[0])
        chosen = []
        for J, x in cand:
            if all(np.abs(x - y).max() > 2e-3 for _, y in chosen):
                chosen.append((J, x))
            if len(chosen) >= a.validate_top:
                break
        V = pool.map(validate, [x for _, x in chosen], chunksize=1)
    checks = [dict(cost=J, fell=f, walk=(not f) and m_.get("gait") == "walk" and bool(m_.get("alternating"))
                   and m_.get("step_time_cv", 1.0) < 0.05)
              for (J, _), (f, m_) in zip(chosen, V)]
    ok = [i for i, c in enumerate(checks) if c["walk"]]
    print("final selection: %d of %d candidates walk steadily for 30 s" % (len(ok), len(checks)), file=log, flush=True)
    i_sel = ok[0] if ok else 0
    best = (chosen[i_sel][0], chosen[i_sel][1], best[2] if i_sel == 0 else {})
    m = V[i_sel][1]
    if "dlf" in m:
        m["dlf1"] = m["dlf"][0]
    p = to_params(best[1], a.g)
    res = dict(kind="tuned", similarity_search=not a.no_similarity, g_gain=a.g_gain if a.g_gain else a.g,
               g=a.g, target_speed=a.speed, targets={k: list(v) for k, v in tg.items()}, leg=a.leg,
               x01=[float(v) for v in best[1]], free=FREE, cost=float(best[0]),
               walker="3d", seed_from=seed_files[1], seed_gait=seed_files[0], fixed=FIXED,
               bounds=dict(lo=LO.tolist(), hi=HI.tolist()),
               params=p.asdict(), metrics_opt=best[2], metrics_validation_30s=m,
               seed=a.seed, popsize=a.popsize, generations=len(hist), sigma0=a.sigma,
               history=hist, wall_seconds=time.time() - t_start,
               final_selection=dict(checked=checks, chosen_rank=int(i_sel), any_stable=bool(ok)),
               evaluation=dict(T=a.T, t_skip=a.t_skip, dt=DT))
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("validation 30 s (chosen rank %d):" % i_sel, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in m.items() if not isinstance(v, list)}), file=log, flush=True)


if __name__ == "__main__":
    main()
