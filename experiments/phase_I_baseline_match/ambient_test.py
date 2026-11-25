"""Simulated ambient vibration test of the deck and its reference modes,
following the earlier study, Sections 2.5 and 2.6.1 step by step.

* the deck alone, from rest, under the three shakers (800 N per component plus
  the 5 N vertical floor, same layout and spectrum as in every trial);
* 30 channels: triaxial acceleration at 0.20, 0.30, 0.50, 0.70 and 0.80 L on
  both edges of the walking surface, each with independent 30 ug/sqrt(Hz)
  white noise (the baseline's noise generator and seeds);
* length max(120 s, 1000 periods of B1) = 554.85 s; the same modal deck,
  integrator (RK4) and step (0.25 ms) as the trials;
* decimation 1000 -> 125 Hz, cross-spectral matrix (4096-sample Hann windows,
  50 % overlap), first singular value of all 30 channels (bending, torsion)
  and of the 10 lateral channels (lateral modes);
* reference frequencies and half-power bands by the baseline's own functions
  (vendored verbatim: benchmark_references).

Writes outputs/tables/reference_modes.json; prints a comparison with the
earlier study's Table 3.

    python experiments/phase_I_baseline_match/ambient_test.py
"""
import json
import os
import time

import numpy as np

from planetsense.evaluation import legacy_pipeline as LP
from planetsense.excitation.multisine import make_baseline_shakers, shaker_layout
from planetsense.simulation.records import DT
from planetsense.structure.spine_beam import SpineBeamDeck

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHAKER_RMS, FLOOR_RMS, BENCH_SEED = 800.0, 5.0, 11
# the earlier study, Table 3 (baseline deck): reference frequency, prominence (dB), band (Hz); None = not a reference
TABLE3 = {"B1": (1.8024, 24.80, (1.7650, 1.8329)), "B2": (7.2101, 16.73, (7.1281, 7.3358)),
          "T1": (12.2911, 17.05, (11.9098, 12.6221)), "B3": (16.1523, 5.85, (15.7645, 16.5609)),
          "L1": (22.4389, 19.39, (22.0076, 22.6826)), "T2": (24.8663, 15.71, (24.3934, 25.1996)),
          "B4": None, "T3": (37.2076, 6.42, (36.9920, 38.0416)), "B5": None, "T4": None}


def simulate(deck, T, fs_out=1000.0):
    """Deck alone from rest; returns accelerations (n, 30) at fs_out and the channel list."""
    L, hw = deck.params.L, 0.5 * deck.params.W
    chans = [(fr, sd, a) for fr in (0.20, 0.30, 0.50, 0.70, 0.80) for sd in (-1, 1) for a in range(3)]
    xy = np.array([[fr * L, sd * hw] for fr, sd, _ in chans])
    S = deck.surface_modes(xy[:, 0], xy[:, 1])                        # (30, 3, nm)
    Sch = np.array([S[i, a] for i, (_, _, a) in enumerate(chans)])    # (30, nm)
    sxy = shaker_layout(L, deck.params.W)
    Ssh = deck.surface_modes(sxy[:, 0], sxy[:, 1]).reshape(-1, deck.nm).T   # (nm, 9)
    forcing = make_baseline_shakers(T=T + 4.0, rms=SHAKER_RMS, floor_rms=FLOOR_RMS, seed=BENCH_SEED)
    n = int(round(T / DT))
    every = int(round(1.0 / (fs_out * DT)))
    k, c = deck.omega ** 2, 2 * deck.zeta * deck.omega
    eta, etad = np.zeros(deck.nm), np.zeros(deck.nm)
    Q = Ssh @ forcing.sample(0.5 * DT, 2 * n + 1, 0.0)               # modal forces at the half steps
    A = np.zeros((n // every, deck.nm))
    io = 0
    for i in range(n):
        q0, qh, q1 = Q[:, 2 * i], Q[:, 2 * i + 1], Q[:, 2 * i + 2]
        a1 = q0 - c * etad - k * eta
        if i % every == 0 and io < len(A):
            A[io] = a1                                                  # output at the start of a step, as in simulate()
            io += 1
        e2, v2 = eta + 0.5 * DT * etad, etad + 0.5 * DT * a1
        a2 = qh - c * v2 - k * e2
        e3, v3 = eta + 0.5 * DT * v2, etad + 0.5 * DT * a2
        a3 = qh - c * v3 - k * e3
        e4, v4 = eta + DT * v3, etad + DT * a3
        a4 = q1 - c * v4 - k * e4
        eta = eta + DT / 6.0 * (etad + 2 * v2 + 2 * v3 + v4)
        etad = etad + DT / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    return A @ Sch.T, chans


def main():
    t0 = time.time()
    deck = SpineBeamDeck().modal(f_max=150.0)
    cfg = dict(LP.BASE_CONFIG)
    f1 = max(float(deck.freq_hz[0]), 0.2)
    T = max(120.0, 1000.0 / f1)                       # 554.85 s for the baseline deck
    fs = 1000.0
    Y, chans = simulate(deck, T, fs)
    # instrument noise, 30 ug/sqrt(Hz) per channel, the baseline's generator and seeds
    for j in range(Y.shape[1]):
        Y[:, j] = Y[:, j] + LP._white_noise(len(Y), fs, cfg["bench_noise"], int(cfg["bench_seed"]) + 991 + j)
    # processing of run_benchmark_avt (human_di.py), step by step
    dec = max(1, int(round(fs / max(50.0, 2.5 * cfg["psd_fmax"]))))
    ch = [LP._decimate(Y[:, j], dec) for j in range(Y.shape[1])]
    fsd = fs / dec
    npseg = int(2 ** round(np.log2(max(64.0, float(cfg.get("bench_seg_s", 41.0)) * fsd))))
    npseg = int(min(npseg, len(ch[0])))
    npseg = min(npseg, 2 ** int(np.floor(np.log2(max(len(ch[0]) // 10, 16)))))
    f, G = LP._csd_matrix(ch, fsd, npseg, npseg // 2)
    m = (f >= cfg["psd_fmin"]) & (f <= cfg["psd_fmax"])
    f, G = f[m], G[m]
    sv1 = np.linalg.svd(G, compute_uv=False)[:, 0]
    sv1_axis = {}
    for a, ax in enumerate("xyz"):
        idx = [kk for kk, cc in enumerate(chans) if cc[2] == a]
        sv1_axis[ax] = np.linalg.svd(G[:, idx][:, :, idx], compute_uv=False)[:, 0]
    band = [(lab, float(fr)) for lab, fr in zip(deck.labels, deck.freq_hz) if cfg["psd_fmin"] <= fr <= cfg["psd_fmax"]]
    bench = dict(f=f, sv1=sv1, sv1_axis=sv1_axis, fe_freqs=[fr for _, fr in band], mode_labels=[lab for lab, _ in band])
    br = LP.benchmark_references(bench, cfg)
    n_avg = 1 + max(0, (len(ch[0]) - npseg) // (npseg // 2))
    res = dict(
        description="Simulated ambient vibration test of the deck (the earlier study, Sections 2.5, 2.6.1)",
        T_s=T, fs_hz=fs, decimated_hz=fsd, nperseg=npseg, n_averages=n_avg, df_hz=float(f[1] - f[0]),
        shaker_rms_N=SHAKER_RMS, floor_rms_N=FLOOR_RMS, channel_noise_ug=cfg["bench_noise"],
        rms_m_s2={ax: float(np.std(Y[:, [kk for kk, cc in enumerate(chans) if cc[2] == a]])) for a, ax in enumerate("xyz")},
        modes=[dict(label=r["label"], f_fe=r["f_fe"], curve=r.get("curve"), f_meas=r["f_meas"], prom_db=r["prom_db"],
                    reference=bool(r["observable"]), reason=r["reason"]) for r in br["assess"]],
        refs=[[lab, fr] for lab, fr in br["refs"]], bands=[list(b) for b in br["bands"]], curves=br["curves"],
        sv1=dict(f=[round(float(v), 5) for v in f], all=[float(f"{v:.5g}") for v in sv1],
                 lateral=[float(f"{v:.5g}") for v in sv1_axis["y"]]),
        wall_seconds=time.time() - t0)
    json.dump(res, open(os.path.join(ROOT, "outputs", "tables", "reference_modes.json"), "w"), indent=1)
    print(f"T = {T:.2f} s, fs {fsd:g} Hz after decimation, {n_avg} averages, df = {f[1] - f[0]:.4f} Hz, "
          f"{time.time() - t0:.0f} s")
    print("| Mode | f_FE (Hz) | curve | f_ic here (Hz) | earlier study's Table 3 | prominence here / earlier study (dB) | band here | band earlier study |")
    print("|---|---|---|---|---|---|---|---|")
    refd = dict((lab, (fr, b)) for (lab, fr), b in zip(br["refs"], br["bands"]))
    for r in br["assess"]:
        t3 = TABLE3.get(r["label"])
        here = refd.get(r["label"])
        print(f"| {r['label']} | {r['f_fe']:.4f} | {r.get('curve')} | "
              + (f"{here[0]:.4f}" if here else f"({r['f_meas']:.4f})" if r["f_meas"] else "–") + " | "
              + (f"{t3[0]:.4f}" if t3 else "not a reference") + " | "
              + f"{r['prom_db']:.2f} / " + (f"{t3[1]:.2f}" if t3 else "–") + " | "
              + (f"{here[1][0]:.4f}–{here[1][1]:.4f}" if here else "–") + " | "
              + (f"{t3[2][0]:.4f}–{t3[2][1]:.4f}" if t3 else "–") + " |")


if __name__ == "__main__":
    main()
