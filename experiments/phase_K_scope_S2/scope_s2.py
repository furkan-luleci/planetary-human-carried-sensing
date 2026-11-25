"""Phase K scope (2026-11-21): design check of a second, habitat-like structure (S2)
before any trial is run.  S2 is a floor of the NASA Common Habitat's mid deck built as the
Habitat Demonstration Unit's floor and sized to its documented loads; see s2_deck.py.

1. S2's modes follow from the documented design (nothing is tuned); their distance from
   the step harmonics of the six main gaits is reported for S1 and S2.
2. Modes of S2 and S1 from the spine-beam model.
3. Excitation scaling: the shaker level that gives the EMPTY S2 the same RMS vertical
   acceleration at its standing point as the empty S1 has at its standing point under
   the earlier study's 800 N per component (frequency-domain, from the shaker spectrum).
4. Human-structure interaction preview: a standing person (nominal calibrated body) on
   S1 and S2 under each gravity, linearised about equilibrium; driving-point frequency
   response under the feet, empty against occupied, and the eigenvalues near the first
   mode.  Linear and small-amplitude only; the time-domain trials would be the test.

Writes outputs/tables/scope_s2.json and outputs/figures/scope_s2.png.

    PYTHONPATH=src python experiments/phase_K_scope_S2/scope_s2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import s2_deck as S2  # noqa: E402
from planetsense.evaluation.linear import frf  # noqa: E402
from planetsense.excitation.multisine import shaker_layout  # noqa: E402
from planetsense.human.body_v2 import load_leg  # noqa: E402
from planetsense.simulation.standing import make_standing  # noqa: E402
from planetsense.structure.spine_beam import SpineBeamDeck  # noqa: E402

G = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}
MAIN = ["earth_1.00", "mars_1.00", "moon_1.00", "earth3d_1.00", "mars3d_1.00", "moon3d_1.00"]
OTHER = ["earth_1.25", "mars_0.77_similar", "moon_0.51_similar", "earth_1.50"]
S1_STAND, S1_Y = (4.2, 0.575), 0.575
S2_Y = 0.0                                     # walking and standing on a beam line (the strip's centreline)
SHAKER_S1, FLOOR_S1 = 800.0, 5.0


def step_freqs():
    return {c: json.load(open(os.path.join(ROOT, "outputs", "gait", c + ".json")))["metrics_validation_30s"]["step_frequency"]
            for c in MAIN + OTHER}


def clearance(f, fs, conds, div=1):
    return float(min(min(abs(f - k * fs[c] / div) for k in range(1, 60)) for c in conds))


def empty_acc_frf(md, x_in, y_in, d_in, x_out, y_out, f):
    """Vertical acceleration at (x_out, y_out) per unit force along axis d_in at (x_in, y_in)."""
    b = md.surface_modes([x_in], [y_in])[0, d_in, :]
    c = md.surface_modes([x_out], [y_out])[0, 2, :]
    w = 2 * np.pi * np.asarray(f)[:, None]
    H = (c * b)[None, :] / (md.omega[None, :] ** 2 - w ** 2 + 2j * md.zeta[None, :] * md.omega[None, :] * w)
    return -(w[:, 0] ** 2) * H.sum(axis=1)


def shaker_rms_response(md, x_out, y_out, rms, floor_rms, T=200.0):
    """RMS vertical acceleration at (x_out, y_out) under the three shakers (x, y, z channels of
    RMS ``rms`` each, plus a vertical floor of ``floor_rms`` per point), from the shaker spectrum
    (flat to 75 Hz, 4th-order Butterworth roll-off to 150 Hz, lines from 0.2 Hz)."""
    k = np.arange(int(np.ceil(0.2 * T)), int(np.floor(150.0 * T)) + 1)
    f = k / T
    prof = 1.0 / np.sqrt(1.0 + (f / 75.0) ** 8)
    unit = prof / np.sqrt(0.5 * np.sum(prof ** 2))          # line amplitudes for RMS 1
    tot = 0.0
    for (xs, ys) in shaker_layout(md.params.L, md.params.W):
        for d, r in ((0, rms), (1, rms), (2, np.hypot(rms, floor_rms))):
            H = empty_acc_frf(md, xs, ys, d, x_out, y_out, f)
            tot += 0.5 * np.sum((r * unit) ** 2 * np.abs(H) ** 2)
    return float(np.sqrt(tot))


def occupied(md, g, x, y, f, leg):
    sc = make_standing(g=g, deck=md, leg=leg, x_com=x, y_com=y, shakers=False)
    s = sc.system
    A, _, f0 = s.linearize(sc.y_eq)
    nq, nm, nx = s.nq, s.nm, A.shape[0]
    b = md.surface_modes([x], [y])[0, 2, :]
    B = np.zeros((nx, 1)); B[2 * nq + nm:2 * nq + 2 * nm, 0] = b
    C = np.zeros((1, nx)); C[0, 2 * nq:2 * nq + nm] = b
    H = frf(A, B, C, np.zeros((1, 1)), f)[:, 0, 0] * -(2 * np.pi * f) ** 2
    lam = np.linalg.eigvals(A)
    return H, lam, float(np.abs(f0).max())


def peaks(f, H, lo, hi):
    a = np.abs(H)
    i = [j for j in range(1, len(f) - 1) if lo <= f[j] <= hi and a[j] >= a[j - 1] and a[j] >= a[j + 1]]
    out = []
    for j in i:
        half = a[j] / np.sqrt(2)
        l = j
        while l > 0 and a[l] > half:
            l -= 1
        r = j
        while r < len(f) - 1 and a[r] > half:
            r += 1
        ok = l > 0 and r < len(f) - 1          # both half-power points inside the band
        out.append(dict(f=float(f[j]), height=float(a[j]),
                        zeta_half_power=float((f[r] - f[l]) / (2 * f[j])) if ok else None))
    return out


def near_eig(lam, f0, band=0.35):
    lam = lam[lam.imag > 1e-6]
    fr = np.abs(lam) / (2 * np.pi)
    sel = np.abs(fr - f0) < band * f0
    return [dict(f=float(np.abs(l) / (2 * np.pi)), zeta=float(-l.real / abs(l))) for l in lam[sel][np.argsort(fr[sel])]]


def main():
    fs = step_freqs()
    h = S2.HabitatDeck()
    deck2 = S2.build(h)
    md2 = deck2.modal(f_max=150.0)
    des = deck2.meta["design"]
    md1 = SpineBeamDeck().modal(f_max=150.0)
    x2 = 0.25 * md2.params.L                   # middle of the first span
    out = dict(status="SCOPE ONLY - design check before any S2 trial; linear analysis, nominal body",
               deck=dict(inputs={k: v for k, v in h.__dict__.items()}, design=des, total_mass=float(des["total_mass_strip"]),
                         sources=["Howard 2021, NASA NTRS 20210021782 (Common Habitat)",
                                  "Smith & Langford 2012, NASA NTRS 20120008160 (HDU floor)",
                                  "Yin 1989, NASA NTRS 19900015683 (lunar habitat interior framing)",
                                  "AISC Design Guide 11 (effective width, joist mode)"]),
               standing_point=[x2, S2_Y], walking_line_y=S2_Y,
               step_frequency=fs)
    modes = lambda md: [dict(label=l, f=float(fq), zeta=float(z)) for l, fq, z in zip(md.labels, md.freq_hz, md.zeta)]
    out["modes_S2"], out["modes_S1"] = modes(md2), modes(md1)
    out["clearance"] = {
        name: [dict(label=m["label"], f=m["f"], step_main=clearance(m["f"], fs, MAIN), step_all=clearance(m["f"], fs, MAIN + OTHER),
                    stride_main=clearance(m["f"], fs, MAIN, 2)) for m in ms if m["f"] < 30]
        for name, ms in (("S1", out["modes_S1"]), ("S2", out["modes_S2"]))}
    r1 = shaker_rms_response(md1, *S1_STAND, SHAKER_S1, FLOOR_S1)
    r2u = shaker_rms_response(md2, x2, S2_Y, 1.0, FLOOR_S1 / SHAKER_S1)
    scale = r1 / r2u
    out["excitation"] = dict(rule="empty S2 at its standing point = empty S1 at its standing point under 800 N",
                             S1_rms_acc_empty=r1, S2_shaker_rms=float(scale), S2_floor_rms=float(scale * FLOOR_S1 / SHAKER_S1),
                             S2_check_rms_acc=shaker_rms_response(md2, x2, S2_Y, scale, scale * FLOOR_S1 / SHAKER_S1))
    leg = load_leg()
    f = np.linspace(0.5, 25.0, 4901)
    hsi = {}
    for name, md, (x, y), fm in (("S2", md2, (x2, S2_Y), md2.freq_hz[0]), ("S1", md1, S1_STAND, md1.freq_hz[0])):
        H0 = empty_acc_frf(md, x, y, 2, x, y, f)
        hsi[name] = dict(f=f.tolist(), empty=np.abs(H0).tolist(), empty_peaks=peaks(f, H0, 0.5 * fm, 2.2 * fm))
        for p, g in G.items():
            H, lam, res = occupied(md, g, x, y, f, leg)
            hsi[name][p] = dict(abs=np.abs(H).tolist(), peaks=peaks(f, H, 0.5 * fm, 2.2 * fm), eig_near_B1=near_eig(lam, fm),
                                eq_residual=res)
            print(name, p, "peaks", [(round(q["f"], 3), None if q["zeta_half_power"] is None else round(q["zeta_half_power"], 4)) for q in hsi[name][p]["peaks"]],
                  "| eigen near B1", [(round(e["f"], 3), round(e["zeta"], 4)) for e in hsi[name][p]["eig_near_B1"]], flush=True)
    out["hsi"] = hsi
    os.makedirs(os.path.join(ROOT, "outputs", "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "scope_s2.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("excitation",)}, indent=1))
    print("S2:", [(m["label"], round(m["f"], 2)) for m in out["modes_S2"] if m["f"] < 35], "| mass", round(out["deck"]["total_mass"]))
    print("clearance", json.dumps(out["clearance"], indent=0)[:400])
    figure(out, fs)


def figure(out, fs):
    """Scope figure in the project's plate style (skipped when that style module is not available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import plate_style as PS
    except ImportError:
        return
    PS.apply()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.0], hspace=0.62, wspace=0.28, left=0.135, right=0.98, top=0.93, bottom=0.11)
    a = fig.add_subplot(gs[0, :])
    lo = min(fs[c] for c in MAIN); hi = max(fs[c] for c in MAIN)
    for k in range(1, 18):
        a.axvspan(k * lo, k * hi, color=PS.COL["ochre"], alpha=0.22, lw=0, zorder=0)
    for row, (name, ms) in enumerate((("S1", out["modes_S1"]), ("S2", out["modes_S2"]))):
        yv = 1 - row
        for m in ms:
            if m["f"] > 30:
                continue
            a.plot([m["f"]] * 2, [yv - 0.28, yv + 0.28], color=PS.COL["ink"], lw=1.4, solid_capstyle="butt")
            a.text(m["f"], yv + 0.34, m["label"], ha="center", va="bottom", fontsize=6.4, family=PS.MONO)
    a.set_yticks([1, 0]); a.set_yticklabels(["S1 footbridge\n20.2 t", "S2 habitat deck\n%.2f t (strip)" % (out["deck"]["total_mass"] / 1000)])
    a.set_ylim(-0.6, 1.75); a.set_xlim(0, 30); a.set_xlabel("Frequency (Hz)")
    PS.stamp(a, "shaded: step harmonics of the six main gaits (%.2f\u2013%.2f steps/s)" % (lo, hi), y=0.04)
    PS.tag(a, "a")
    for j, (name, lo_f, hi_f) in enumerate((("S2", 5.0, 22.0), ("S1", 1.0, 9.0))):
        b = fig.add_subplot(gs[1, j])
        h = out["hsi"][name]
        f = np.asarray(h["f"]); m = (f >= lo_f) & (f <= hi_f)
        b.semilogy(f[m], np.asarray(h["empty"])[m], color=PS.COL["ink"], lw=0.9, ls=(0, (1, 1.6)), label="empty")
        for p in G:
            b.semilogy(f[m], np.asarray(h[p]["abs"])[m], **PS.planet_line_kw(p), label=p + ", person standing")
        b.set_xlim(lo_f, hi_f); b.set_xlabel("Frequency (Hz)")
        b.set_ylabel("Under-foot acceleration\nper newton (m s$^{-2}$ N$^{-1}$)" if j == 0 else "")
        PS.tag(b, "b" if j == 0 else "c")
        PS.stamp(b, name + (" (strip, %.0f kg)" % out["deck"]["total_mass"] if name == "S2" else " (20 160 kg)"), y=0.04)
        if j == 0:
            b.legend(loc="upper left", fontsize=6.0, frameon=False)
    os.makedirs(os.path.join(ROOT, "outputs", "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "scope_s2.png"), dpi=200, facecolor="white")
    print("figure written")


if __name__ == "__main__":
    main()
